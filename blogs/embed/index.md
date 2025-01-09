#! Text Embedding at Scale

*by [Doug Hanley](/) [[@iamlemec](https://twitter.com/iamlemec)], published 2023-11-05*

So I've been getting into text embeddings. In particular, running embedding models fast and at scale. My main goal here is being able to do fast semantic search over very large corpora such as Wikipedia.

There's been a lot of focus on fast inference on large transformer models such as Llama and friends. Because most embedding models nowadays are based on transformers and are produced by averaging over their latent space representations ^[That is, if your latent space is of size `[T,D]`, where `T` is the sequence length and `D` is the dimensionality, we can average over the `T` dimension approporiately to get the embedding for a given sample.], a lot of these tools and techniques are portable.

In this blog post, I'm going to go through various embedding optimization techniques, talk a bit about vector databases and quantization, and then do some benchmarks on various embedding models and existing vector databases. *If you don't have time to read the whole post, my main takeaways are:*

- With aggressive optimization, we can embed large corpora like Wikipedia in ~1 hour
- GPUs still have a huge advantage over CPUs at embedding, even the slightly older GPUs
- Smaller models like `bge-micro-v2` are quite fast and tend to be tokenizer limited
- OpenAI embeddings like `text-embedding-ada-002` are very slow due to rate limits
- Going down to 4-bit quantization entails no degredation in performance
- With some relatively minor changes, Chroma could be a lot faster at embedding

In the process of looking into all of this, I ended up writing a pretty fast quantization aware vector database on top of `torch`, which I'm cally `ziggy`. It has custom `torch` kernels (both CPU and GPU) for quantization, dequantization, and quantized matrix multiplication for similarity search. It also has a wrapper for `huggingface` embeddings that gets you fully optimized and parallelized performance. If you want to check that out, head over to: https://github.com/iamlemec/ziggy. It's still under active development, and any contributions are very welcome.

Have comments, corrections, or suggestions? Drop me a line on Twitter [@iamlemec](https://twitter.com/iamlemec).

#* Making Things Go Fast

There's a model optimization framework from Microsoft called ONNX that can really speed up inference here. It performs optimizations such as operator fusion, representing the result in an intermediate format, and can execute the optimized model on CPU/GPU runtimes.

Since we have these fancy GPUs at our disposal, we want to be sure we're making the most of them, and that means maximizing utilization. When you get the proper embedding part fast enough, you end up in a situation where tokenization is a non-trivial fraction of the compute time. But because tokenization is done on the CPU, you can run tokenization and embedding in a parallel pipeline, with tokenization on the CPU and embedding on the GPU. This can bring utilization from around 45% to roughly 80% in my experience.

After these optimizations, your main determinants of embed speed will be the size of the model you're using and the maximum sequence length you allow. The most common embedding model nowadays seems to be `all-MiniLM-L6-v2` which comes as part of the `sentence-transformers` library. It has pretty good runtime and performance characteristics and situates you on a nice spot along the speed/accuracy Pareto frontier ^[I like how this term has found its way from economics to computer science over the years. We typically only use it in multi-agent settings, but I suppose you could imagine a world consisting of two people, one who only cares about speed and one who only cares about accuracy]. There's a regularly updated [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) that evaluates different models across a wide variety of benchmarks.

The `all-MiniLM-L6-v2` encoder model has about 10 million parameters (quite small by LLM standards, which run in the billions), a maximum sequence length of 512 tokens, and outputs a 384 dimensional vector embedding. Most of the models I look at in this post are pretty similar in terms of token length and embedding dimensionality. The main point of variation is the parameter count.

#* Vector Databases and Quantization

There are a million framework's for storing and querying these vectors. Being unable to choose a specific one, I decided to write my own and learn what I could along the way. The main focus was on speed and memory footprint. Wikipedia has about 37 million distinct paragraphs across 6 million articles. Storing these at half precision (16 bits) for a typical embedding (with `D=384`) requires around 26GB of memory, which exceeds the memory capacity of most any consumer GPU. You have to go up into the A6000/A100 range to fit that.

So we need to compress this thing. That's where *quantization* comes in. By laying out a grid over an appropriate range, you can instead round a given floating point number to the nearest grid point and index it using an integer. Thus 8-bit quantization allows for 256 possible values, 4-bit for 16 values, and so on. You can even go down to 1-bit optimization, which is essentially just tracking the sign of each component of the embedding, and it actually still yields halfway decent results.

Once the documents are embedded and stored with the desired quantization, we'll want to query the vector database to find relevant documents. To do this, we compute the *cosine similarity* between the embedding of a given query (say "World War II") and each vector in our database. The cosine similarity between two $\ell^2$-normalized vectors is

$$* CS(v_1,v_2) = v_1 \cdot v_2 = \sum_k v_{1,k} v_{2,k} \in [-1, 1]

It's a little inefficient to compute millions of similarities for one lousy query, but it's actually doable in about 500ms at Wikipedia scale. If we want to shorten this to reduce latency, we can group our sentences into documents by averaging (storing these in a new smaller index) and do a hierarchical query. This gets it down to about 50ms per query. Other vector databases such as `faiss` use more sophisticated methods such as product quantization or K-means clustering to reduce query runtime.

#* Taking Down Some Datasets

Because some of my research focuses on analyzing Wikipedia, I've focused on that as my main target for benchmarking. Also, it's generally useful and something people would naturally want to index and search through. Below is a speed-performance chart for some notable or high-ranking embedding models. The dot size is proportional to the number of model parameters. I've also included a point for the performance of OpenAI's primary embedding product `text-embedding-ada-002`. While I have no doubt of their internal capabilities, the numbers for this benchmark are purely a result of the API's  rate limit of 1 million tokens per minute.

!gum [width=80]
let data = [
  {"Model":"all-MiniLM-L6-v2","Seqs":6064.483871,"Params":22.7,"MTEB":56.26,"xoff":0.0,"yoff":0.0},
  {"Model":"bge-small-en-v1.5","Seqs":3767.51503,"Params":33.4,"MTEB":62.17,"xoff":0.0,"yoff":0.0},
  {"Model":"bge-base-en-v1.5","Seqs":1516.120968,"Params":109,"MTEB":63.55,"xoff":70,"yoff":-0.02},
  {"Model":"bge-micro-v2","Seqs":8909.905213,"Params":17.4,"MTEB":56.57,"xoff":0.0,"yoff":0.0},
  {"Model":"openai-ada-002","Seqs":132.486963,"Params":10,"MTEB":60.99,"xoff":-30,"yoff":0.0}
];
let [xmin, xmax] = [0, 10000]; let [ymin, ymax] = [55, 65];
let [model, seqs, size, mteb, xoff, yoff] = zip(...data.map(d =>
  [d["Model"], d["Seqs"], d["Params"], d["MTEB"], d["xoff"], d["yoff"]]
));
let points = Points(zip(seqs, mteb, size).map(([s, m, k]) =>
  [Dot(), [s, m], [120*sqrt(k)/10, sqrt(k)]]
));
let labels = Points(zip(model, seqs, mteb, xoff, yoff).map(([m, x, y, xo, yo]) =>
  [Anchor(Text(m), {align: 'right'}), [x+350+xo, y+0.03+yo]]
), {size: [160, 10]});
let adaline = HLine(60.99, {stroke_dasharray: 4, lim: [xmin, xmax]});
let adalab = Place(Text('text-embedding-ada-002'), {pos: [8400, 60.6], rad: [1500, 10]});
let plot = Plot([points, labels], {
  aspect: phi, xlim: [xmin, xmax], ylim: [ymin, ymax], yticks: 5,
  xlabel: 'Sequences / Second', ylabel: 'MTEB Score', title: 'Speed vs Performance',
  xlabel_offset: 0.15, ylabel_offset: 0.12, title_offset: 0.05,
  xlabel_font_size: 17, ylabel_font_size: 17, title_size: 0.09
});
let frame = Frame(plot, {padding: [0.25, 0.15, 0.15, 0.2]});
return frame;

So the various `bge-*` models span the Pareto frontier from quite fast to extremely fast, while `all-MiniLM-L6-v2` strictly dominated by `bge-micro-v2`. For comparison to some non-dedicated GPU numbers, the folks as bloop have done some optimized speed benchmarks with an M2 Max [here](https://bloop.ai/blog/gpu_with_ggml). Their numbers for `all-MiniLM-L6-v2` (with token length 256, as here) are coming in around 263 sequences per second, so it seems pure GPU still has a big advantage. To get an idea of the speed characteristics on a slightly older model GPU, I ran these on my RTX 2080 and found sequence per second numbers of about 4500 for `bge-micro-v2` and around 3000 seq/sec for `all-MiniLM-L6-v2`, so still around 10x faster than the M2 numbers. As for pure CPU numbers, I get around 190 seq/sec on a 24 core Ryzen 9 7900X.

These benchmarks were on a subset of about 5000 random Wikipedia articles. Each article is broken up into paragraphs, which are then embedded individually. Paragraphs that are longer than the maximum token length are broken up, embedded, then averaged. In all, there are about 6.6 million English Wikipedia articles, for a total of 37 million paragraphs and about 2.5 billion tokens. Extrapolating these numbers to a full embedding of the Wikipedia corpus, we can get an idea of how long that takes.

I've actually run it for a couple of different configurations, and the numbers line up as you would expect. Embedding all of Wikipedia using ` bge-micro-v2` takes about 1 hour on an A6000, so even for the bulkier `bge-base-en-v1.5`, you could get it done in less than 5 hours. At this scale, the total size of the embedding output starts to matter, especially if you want to fit them on scarce GPU memory, so quantization is also very important here. Using 4-bit quantization, the memory footprint of a `D=384` embedding is 7.7GB. One could bring this all the way down to 1.9GB using 1-bit quantization or forego quantization altogether and store it in half precision at 31GB.

So let's look at what happens when you start compressing with quantization. Here, for the various models (at 256 max token length) I've run the [MTEB](https://github.com/embeddings-benchmark/mteb) retrieval benchmarks at different levels of quantization, namely half precision (16 bit) floating point and 8/4/2/1-bit integer quantization ^[Note that the previous scores were for all MTEB benchmarks, so they aren't directly comparable the ones here which are retrieval only. I'm mostly focused on retrieval anyway, and running all the benchmarks would have been too time-consuming. Finally, note that I also don't run the `MSMARCOv2` benchmark, as my measly 128GB of RAM wasn't enough.]. With this we can plot these scores versus the amount of memory needed for storage and get an idea of what the Pareto frontier looks like.

!gum [width=80]
let pal = x => interpolateHex('#1e88e5', '#ff0d57', x);
let data = [
  {"Short":"bge-base","Color":1,"Bits":1,"MTEB":43.590584,"Memory":96,"xoff":0,"yoff":-0.25},
  {"Short":"bge-base","Color":1,"Bits":2,"MTEB":45.498558,"Memory":192,"xoff":0,"yoff":0},
  {"Short":"bge-base","Color":1,"Bits":4,"MTEB":49.46743,"Memory":384,"xoff":0,"yoff":0},
  {"Short":"bge-base","Color":1,"Bits":8,"MTEB":49.750866,"Memory":768,"xoff":0,"yoff":0},
  {"Short":"bge-base","Color":1,"Bits":16,"MTEB":49.552265,"Memory":1536,"xoff":0,"yoff":0},
  {"Short":"bge-small","Color":2/3,"Bits":1,"MTEB":38.619087,"Memory":48,"xoff":0,"yoff":0},
  {"Short":"bge-small","Color":2/3,"Bits":2,"MTEB":44.030624,"Memory":96,"xoff":0,"yoff":0.3},
  {"Short":"bge-small","Color":2/3,"Bits":4,"MTEB":47.4528,"Memory":192,"xoff":0,"yoff":0},
  {"Short":"bge-small","Color":2/3,"Bits":8,"MTEB":47.51902,"Memory":384,"xoff":0,"yoff":0},
  {"Short":"bge-small","Color":2/3,"Bits":16,"MTEB":47.118233,"Memory":768,"xoff":0,"yoff":0},
  {"Short":"miniLM","Color":1/3,"Bits":1,"MTEB":35.934536,"Memory":48,"xoff":0,"yoff":0},
  {"Short":"miniLM","Color":1/3,"Bits":2,"MTEB":39.98356,"Memory":96,"xoff":0,"yoff":0},
  {"Short":"miniLM","Color":1/3,"Bits":4,"MTEB":41.650852,"Memory":192,"xoff":0,"yoff":0.2},
  {"Short":"miniLM","Color":1/3,"Bits":8,"MTEB":41.585796,"Memory":384,"xoff":0,"yoff":0.25},
  {"Short":"miniLM","Color":1/3,"Bits":16,"MTEB":41.745193,"Memory":768,"xoff":0,"yoff":0},
  {"Short":"bge-micro","Color":0,"Bits":1,"MTEB":33.499,"Memory":48,"xoff":0,"yoff":0},
  {"Short":"bge-micro","Color":0,"Bits":2,"MTEB":38.055523,"Memory":96,"xoff":0,"yoff":0},
  {"Short":"bge-micro","Color":0,"Bits":4,"MTEB":40.89632,"Memory":192,"xoff":0,"yoff":-0.2},
  {"Short":"bge-micro","Color":0,"Bits":8,"MTEB":40.925518,"Memory":384,"xoff":0,"yoff":-0.25},
  {"Short":"bge-micro","Color":0,"Bits":16,"MTEB":40.534416,"Memory":768,"xoff":0,"yoff":0},
];
let [xmin, xmax] = [3, 10]; let [ymin, ymax] = [30, 55];
let [short, col, bits, mem, mteb, xoff, yoff] = zip(...data.map(d =>
  [d["Short"], d["Color"], d["Bits"], d["Memory"], d["MTEB"], d["xoff"], d["yoff"]]
));
let mem3 = mem.map(b => log(b/3)/log(2));
let points = Points(zip(mem3, mteb, col, bits).map(([m, t, c, b]) => {
  let stroke = interpolateHex(pal(c), '#000000', 0.5);
  let dot = Dot({fill: pal(c), stroke_width: 0.75, stroke: stroke, opacity: 0.75});
  let size = 0.03*pow(b, 0.3);
  return [dot, [m, t], size];
}));
let labels = Points(zip(short, mem3, mteb, col, bits, xoff, yoff).map(([n, x, y, c, b, xo, yo]) => {
  let size = 0.15*pow(b, 0.03);
  let [x1, y1] = [x+size+0.01+xo, y+yo+0.1];
  let text = Text(`${n}-${b}`, {stroke: pal(c), stroke_width: 0.8});
  return [Anchor(text, {align: 'right'}), [x1, y1]]
}), {size: 0.075});
let plot = Plot([points, labels], {
  aspect: phi, xlim: [xmin, xmax], ylim: [ymin, ymax], xticks: 8, yticks: 6,
  xlabel: 'Bytes Per Record [log2(b/3)]', ylabel: 'MTEB Score [Retrieval]', title: 'Memory vs Performance',
  xlabel_offset: 0.15, ylabel_offset: 0.12, title_offset: 0.05,
  xlabel_font_size: 17, ylabel_font_size: 17, title_size: 0.09
});
return Frame(plot, {padding: [0.25, 0.15, 0.15, 0.2]});

Note that because lower memory usage is better, the Pareto frontier here is up and left. One of the striking trends evident is that performance doesn't degrade at all, and even slightly improves in some cases, going from 16-bit to 8-bit and then to 4-bit. Only going down to 2-bit and 1-bit does performance really start to drop. The Pareto frontier is dominated by `bge-small` and `bge-base`. So these may be optimal if embed speed is of little concern relative to memory, while `bge-micro` shines when embed speed is critical.

#* Comparison To Other Frameworks

There are going to be both feature and speed differences here. My framework `ziggy` is pretty light on general indexing features. The main focus is on querying embeddings by similarity. But it's also pretty fast and has good quantization support. Some of the existing vector database options include: `faiss`, `chroma`, `pinecone`, `milvus`, `weaviate`, `qdrant`, `elastic`, and `pgvector`. It seems like only Chroma gets more into doing the embedding step for you, either locally executed or externally through OpenAI or Cohere.

Benchmarking Chroma using `all-MiniLM-L6-v2`, I find that the speed achieved is 1146 seqs/sec using an A6000 GPU, whereas above I report getting about 6000 seqs/sec with `ziggy`. That's with a max token length of 256, but even with the largest available token lenght of 512, I'm getting 3500 seqs/sec. It seems possible that either Chroma is not using ONNX optimizations to their fullest potential or they aren't parallelizing the CPU based tokenization and the GPU based embedding tasks. They should do that!

#* Some Final Thoughts

I'm pretty excited to go embed some big corpora. I've already done Wikipedia, so I guess things like patents, academic papers, and legislation are next. If people have other ideas here, I'm definitely open to them. Being able to do semantic search is great, but there's also a lot of work going on now with retrieval-augmented-generation (RAG) frameworks. The end goal is to massively increase the amount of information at our fingertips and to integrate this seamlessly into our workflow in a way that boosts productivity and minimizes the sometimes huge losses from context switching.

I also think the multi-lingual possibilites here are really exciting. Meta recently released SeamlessM4T, a translation model that can convert between almost any two text/audio-language endpoints. It's pretty incredible, and honestly I think it's gotten less press than I would have expected. And we can use it for language agnostic embeddings. Imagine being able to search through huge corpora that are written in a language you can't even speak, and then autotranslating the results as you need them. Seamless doesn't seem to work great with ONNX currently, but it shouldn't be too hard to get these issues worked out.

Thanks for reading!
