---
layout: post
title: Multiprocessing Dataloader
date: 2026-02-02 10:45:00
description: Simplified implementation of the pytorch dataloader
tags: pytorch multiprocessing dataloader
categories: pytorch
disqus_comments: true
toc:
  beginning: true
---

For many machine learning problems, loading data can be a bottleneck in the training process. To reduce this wait time and train more efficienctly, it is therefore beneficial to have a number of batches already loaded *whilst the model is being trained*. This is already possible with the PyTorch [DataLoader](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py) class, where you can input `num_workers` for the number of worker processes that simultaneously load in the data. To get an idea of how this class works, and how the python multiprocessing library is used, a simplified version is shown in this post. 

For a problem like this we immediately think of multithreading or multiprocessing. Given that loading data can be quite CPU heavy, especially if the data is transformed as it is being loaded in, we will use the Python multiprocessing library. Omitting many of the key implementation details, an overview of the pipeline is given in the following figure:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/2026-02-03-PyTorch-Dataloader/overview_diagram.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Starting with a PyTorch dataset, this will be passed into our dataloader class, which will spawn N processes. These processes will read in the indices of each batch, say `[12, 14, 16]`, load the batch data, then return the data to the dataloader class which yields it. When designing our multiprocessing dataloader class, we need to consider a few important points:
<!-- It does not, however, come without its problems. Below are a few of the issues that need to be considered when making a multiprocessing dataloader: -->
1. We need to ensure that the processes do not read from the same object simultaneously, which would lead to a race condition. 
2. We do not want duplicate batches.
3. Workers can finish loading a batch at different times, but we want batches to always be loaded in a predictable order. 
4. We want to limit the number of batches that are pre-loaded to some user-defined limit, even if training slows for some reason, so that the usage memory doesn't explode.

Before diving into our multiprocessing dataloader, we will start by defining our dataset, and then find a baseline with a non-multiprocessing dataloader. 

## Python Imports

Load in the necessary libraries:

```python
from torch.utils.data import Dataset
import numpy as np
import time 
from typing import Iterator, Dict
import multiprocessing as mp
```

## Dataset Class

We will use fake data which is created on the fly, chosen such that it takes a non-negligible amount of time to retrieve a single batch. The specific details of this are of course not that important, and any other type of data could easily be used instead. For this, we randmly sample `n_points` in 3D space between $$x,y,z \in [0, 1]$$, then find the k-nearest neighbours for each point. The indices of the k-nearest neighbours are then returned as our data. 

```python
def createFakeData(n_points, k_nearest):
    # get random points in 3D space between 0 and 1
    points = np.random.rand(n_points, 3)

    # compute distance matrix between all points
    dist_matrix = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
    # get k nearest neighbors for each point
    knn_indices = np.argsort(dist_matrix, axis=-1)[:, 1:k_nearest+1] # exclude self, so start at 1
    return knn_indices
```

We can now define our dataset class, which inherits from the PyTorch Dataset class. 
This takes in the number of total samples, `n_samples`, and when we index the dataset, it returns a data sample.


```python
class CustomDataset(Dataset):
    def __init__(self, n_samples: int = 10000):
        # Define the number of samples in the dataset
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # catch case where idx goes above self.n_samples
        if idx >= self.n_samples:
            raise IndexError
        
        # Generate fake data for each sample
        data = createFakeData(200, 16)
        return data
```

Example of this being indexed:
```python
dataset = CustomDataset(n_samples=50000)
sample = dataset[10]
print("sample shape:", sample.shape)
```
```text
sample shape: (200, 16)
```

## Non-multiprocessing Dataloader

As a baseline, we will compare our final multiprocessing dataloader to a simple single-processor dataloader, called `BasicDataLoader`. The full code is given below, and explained after:

```python
# define a very basic dataLoader that just creates data when the iterator is called 
class BasicDataLoader():
    def __init__(self, dataset : Dataset , batch_size: int = 1000):
        self.dataset = dataset        # pass in pytorch dataset
        self.batch_size = batch_size  # number of samples in each batch
        
        # get idxs for batches
        data_idxs = np.arange(len(self.dataset))
        # ignore shuffling for now
        self.batched_data_idxs = [data_idxs[i:i + batch_size] for i in range(0, len(data_idxs), batch_size)]

    def __iter__(self) -> Iterator:
        
        for batch_idxs in self.batched_data_idxs:
            # create the batch of data
            batch_data = [self.dataset[idx] for idx in batch_idxs]
            # convert
            batch_data = np.array(batch_data)
            
            yield batch_data
```

Starting in the `__init__` method, we pass in the pytorch `dataset` and the number of samples in each batch, called the `batch_size`.
The list of indices for each batch is then defined. 
```python
def __init__(...):
        ...
        # get idxs for batches
        data_idxs = np.arange(len(self.dataset))
        # ignore shuffling for now
        self.batched_data_idxs = [data_idxs[i:i + batch_size] for i in range(0, len(data_idxs), batch_size)]
```
Note that, for simplicity, we are not including any shuffling of the batches. 

When looping over the `BasicDataLoader`, the `__iter__` method is called: 
```python
def __iter__(self) -> Iterator:
    for batch_idxs in self.batched_data_idxs:
        # create the batch of data
        batch_data = [self.dataset[idx] for idx in batch_idxs]
        # convert
        batch_data = np.array(batch_data)
        
        yield batch_data
```

For each iteration of this loop, we get the indices for a single batch, `batch_idxs`, loop over these to create the batch, called `batch_data`, convert to an array then yield it.  

#### Speed of Non-multiprocessing Dataloader

With a `batch_size` of 300, we can find the time it takes to loop over every batch:

```python
# create dataset
dataset = CustomDataset(n_samples=50000)
# instantiate the dataloader
dataloader = BasicDataLoader(dataset, batch_size=300)

# get the start time
now = time.time()
n_batches = 0
# loop over, but don't actually do anything with the data
for batch in dataloader:
    n_batches += 1
    continue
print(f"Processed {n_batches} batches in {time.time() - now:.2f} seconds, time per batch {(time.time() - now)/n_batches:.4f} seconds")
```
```text
Processed 167 batches in 142.36 seconds, time per batch 0.8524 seconds
```


## Multiprocessing Dataloader

The most important aspect of creating a multiprocessing dataloader is ensuring the different workers do not try and access the same data at the same time, which would result in race conditions. To overcome this, we will use the [Queue](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue) class in the Python `multiprocessing` library. This class is a first-in-first-out queue, so elements added first will leave before those added later, and it automatically handles multiple processes trying to access it simultaneously: it is process-safe. 

Using this we will create two queues. One is called `task_q`, which holds `[batch_idx, batch_indices]`---where `batch_idx` is the index of that specific batch, and `batch_indices` is the indices of the samples in the batch--and which the worker processes pull from. 
The second is called `out_q`, which holds the data batches that are output from the worker processes, and which the dataloader pulls from. A schematic of this is shown below:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/2026-02-03-PyTorch-Dataloader/full_diagram.svg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

Now we can define the function that each worker process runs:

```python 
_END = ("__END__",)

def _mp_worker(dataset: Dataset, task_q: mp.Queue, out_q: mp.Queue):
    try:
        while True:
            # get the next task from the task_q
            task = task_q.get()

            # check if the process needs to end
            if task == _END:
                break

            # if not, make the batch 
            batch_n, idxs = task
            batch = np.array([dataset[int(i)] for i in idxs])

            # put this in the out_q
            out_q.put((batch_n, batch))

    except Exception as e:
        out_q.put(("__EXC__", repr(e)))
    
    finally:
        # as a double check, when ending, send _END to the out_q
        out_q.put(_END)
```
<!-- On line 1 we have the special token called `_END`, which signifies to the worker to stop.  -->
When the function is run, it starts the `while` loop. This repeatedly loops over and tries to get the next task from the `task_q`. If it is the `_END` token, we break the loop and send an `_END` token to the `out_q`. If this isn't the end, and task is a genuine `[batch_idx, batch_indices]`, then it makes the batch (lines 14-15), and puts the batch in the `out_q` (line 18).  

We can now define our dataloader, which is shown in full below and explained after:

```python 
class MPBatchLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 300,
        num_workers: int = 3,
        prefetch_batches: int = 5,
        mp_start_method: str = "spawn",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches # Number of batches to load in advance
        
        # get idxs for batches
        data_idxs = np.arange(len(self.dataset))
        # ignore shuffling for now
        self.batched_data_idxs = [data_idxs[i:i + batch_size] for i in range(0, len(data_idxs), batch_size)]
        self.n_batches = len(self.batched_data_idxs)
        
        # Multiprocessing setup
        self.ctx = mp.get_context(mp_start_method)             # defines how new processes are started, "spawn" is most compatible
        self.task_q = self.ctx.Queue()                         # unbounded task queue to add tasks=(batch_n, idxs) to, where batch_n is the n'th batch      
        self.out_q = self.ctx.Queue(maxsize=prefetch_batches)  # bounded output queue to get (batch_n, batch) from workers, i.e. to yield from
        self.workers = []                                      # list of worker processes

    def __iter__(self) -> Iterator[np.ndarray]:
        
        n_total = len(self.batched_data_idxs)  # total number of batches
        submit_idx = 0                         # index of next batch to submit to task_q
        next_batch_idx = 0                     # idx of next batch to be yielded
        n_finished_wkrs = 0                    # number of workers finished
        reorder: Dict[int, np.ndarray] = {}    # store out-of-order batches here


        # start workers
        self.workers = [
            self.ctx.Process(target=_mp_worker, args=(self.dataset, self.task_q, self.out_q), daemon=True)
            for _ in range(self.num_workers)
        ]
        for p in self.workers:
            p.start()


        # Submit initial self.prefetch_batches of tasks
        # catch the case where n_total < self.prefetch_batches
        while submit_idx < min(self.prefetch_batches, n_total):
            self.task_q.put((submit_idx, self.batched_data_idxs[submit_idx]))
            submit_idx += 1


        try:
            while next_batch_idx < n_total:
                # get the next completed batch from the out queue
                msg = self.out_q.get()

                # check for worker exception
                if isinstance(msg, tuple) and msg[0] == "__EXC__":
                    raise RuntimeError(f"Worker exception: {msg[1]}")

                # if no expection, get the completed batch
                batch_n, batch = msg
                # add this to the reorder dict, as this may be out of order
                reorder[batch_n] = batch

                # Yield in-order batches as soon as possible.
                while next_batch_idx in reorder:
                    yield reorder.pop(next_batch_idx)
                    next_batch_idx += 1

                    # As we advance, submit one more task to keep the out_q full.
                    if submit_idx < n_total:
                        self.task_q.put((submit_idx, self.batched_data_idxs[submit_idx]))
                        submit_idx += 1

        finally:
            # tell workers to stop (always)
            for _ in range(self.num_workers):
                self.task_q.put(_END)

            # loop over all the remaining items in the out_q until the workers get the _END signals
            # and exit. If you don't do this, the workers may hang trying to put to a full out_q 
            # when the while loop above exits early (e.g. you only iterate over a single batch and then exit) 
            while n_finished_wkrs < self.num_workers:
                msg = self.out_q.get()
                if msg == _END:
                    n_finished_wkrs += 1

            # wait for the workers to exit
            for p in self.workers:
                p.join()
```

Starting with the `__init__` method: 

```python 
    def __init__(...):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches # Number of batches to load in advance
```
we pass in the pytorch dataset, `dataset`, the batch size, `batch_size`, 
the number of workers that simultaneously load the data, `num_workers`, and the number of batches that we want to load in advance, `prefetch_batches`.

Like before, we then create the list of samples indices for each batch:
```python 
    def __init__(...):
        ... 

        # get idxs for batches
        data_idxs = np.arange(len(self.dataset))
        # ignore shuffling for now
        self.batched_data_idxs = [data_idxs[i:i + batch_size] for i in range(0, len(data_idxs), batch_size)]
        self.n_batches = len(self.batched_data_idxs)
```

Now we setup all the multiprocessing objects:
```python
    def __init__(...):
        ... 
        # Multiprocessing setup
        self.ctx = mp.get_context(mp_start_method)             # defines how new processes are started, "spawn" is most compatible
        self.task_q = self.ctx.Queue()                         # unbounded task queue to add tasks=(batch_n, idxs) to, where batch_n is the n'th batch      
        self.out_q = self.ctx.Queue(maxsize=prefetch_batches)  # bounded output queue to get (batch_n, batch) from workers, i.e. to yield from
        self.workers = []                                      # list of worker processes
```
The context `ctx` defines how the new processes are started, and what they copy from the parent process, this includes copying the parent's memory, threads, python interpreter state and so on. A detailed overview of the different methods can be found [here](https://dev.to/imsushant12/python-multiprocessing-start-methods-pools-and-communication-4o6d). In brief: one method is `spawn`, which starts a new python interpreter, re-importing all modules, and objects are passed to it by the parent process via pickling; another is `fork`, which shares a memory state with the parent process, but does not copy over threads. For our case, which might include additional threads from training on a GPU with CUDA, we want these threads available on the worker process, so we use `spawn`. We then use a context `ctx` to ensure that all our cross-process objects are created the same way, enabling communication between them.

With the context, we create two queues: `task_q` which is no limit to number of elements added (i.e. it is unbounded); and `out_q` which we limit to size `prefetch_batches` to stop infinite memory increases. Finally, we make a list called `workers`, which our worker processes will be added to.  

Now we can get onto the juicy bit: the `__iter__` method. The overview of the method is as follows:
1. Create worker processes
2. Submit tasks to `task_q` up to the number of `prefetch_batches`. E.g. submit 20 tasks. 
3. In a while loop, retrieve batches from `out_q`, and store these in the `reorder` dictionary. This is needed because the batches might arrive out of order. 
4. If the next batch is in `reorder`, then yield it and remove it from `reorder`.
5. Every time we yield a batch, add another task to the `task_q`. By limiting what is added to `task_q`, we limit the size of the `out_q`, stopping the memory from exploding. 
6. After all batches have been yielded, or the loop is terminated early, send a stop signal to all the workers, and drain the remaining batches in the `out_q`.  

In the code itself, the method starts by defining the following variables:
```python 
    def __iter__(self) -> Iterator[np.ndarray]:
        n_total = len(self.batched_data_idxs)  # total number of batches
        submit_idx = 0                         # index of next batch to submit to task_q
        next_batch_idx = 0                     # idx of next batch to be yielded
        n_finished_wkrs = 0                    # number of workers finished
        reorder: Dict[int, np.ndarray] = {}    # store out-of-order batches here
```

We then start the worker processes, and append them to list called `workers`:
```python
    def __iter__(self) -> Iterator[np.ndarray]:
        ...

        # start workers
        self.workers = [
            self.ctx.Process(target=_mp_worker, args=(self.dataset, self.task_q, self.out_q), daemon=True)
            for _ in range(self.num_workers)
        ]
        for p in self.workers:
            p.start()
```
Next we submit the first set of tasks, but we catch the case where the number of tasks might be less than the number of `prefetch_batches`, otherwise we would get an index error when trying to retrieve data that doesn't exist in the dataset. 
```python
    def __iter__(self) -> Iterator[np.ndarray]:
        ...
        # Submit initial self.prefetch_batches of tasks
        # catch the case where n_total < self.prefetch_batches
        while submit_idx < min(self.prefetch_batches, n_total):
            self.task_q.put((submit_idx, self.batched_data_idxs[submit_idx]))
            submit_idx += 1
```

Now for the main `while` loop:
```python
        try:
            while next_batch_idx < n_total:
                # get the next completed batch from the out queue
                msg = self.out_q.get()

                # check for worker exception
                if isinstance(msg, tuple) and msg[0] == "__EXC__":
                    raise RuntimeError(f"Worker exception: {msg[1]}")

                # if no expection, get the completed batch
                batch_n, batch = msg
                # add this to the reorder dict, as this may be out of order
                reorder[batch_n] = batch

                # Yield in-order batches as soon as possible.
                while next_batch_idx in reorder:
                    yield reorder.pop(next_batch_idx)
                    next_batch_idx += 1

                    # As we advance, submit one more task to keep the out_q full.
                    if submit_idx < n_total:
                        self.task_q.put((submit_idx, self.batched_data_idxs[submit_idx]))
                        submit_idx += 1
```
On line 4 we take the next bit of data from `out_q`. On lines 7-8 we check if it's an error message from the worker, if so we raise that error. If not, we get the data on line 11, and add it to the `reorder` dict on line 13. We then check if the next batch is in `reorder` on line 16, if so we yield it, and add 1 to the `next_batch_idx`. Finally, if a batch is yield, we add another task to the `task_q` on lines 21-23. 

Finally, if all batches have been yielded, or if the loop is terminated early (such as only looping over one batch as a test), we move to the `finally` block. 

```python
        finally:
            # tell workers to stop (always)
            for _ in range(self.num_workers):
                self.task_q.put(_END)

            # loop over all the remaining items in the out_q until the workers get the _END signals
            # and exit. If you don't do this, the workers may hang trying to put to a full out_q 
            # when the while loop above exits early (e.g. you only iterate over a single batch and then exit) 
            while n_finished_wkrs < self.num_workers:
                msg = self.out_q.get()
                if msg == _END:
                    n_finished_wkrs += 1

            # wait for the workers to exit
            for p in self.workers:
                p.join()
```

On lines 3-4 we send the `_END` token to the workers which, looking back on the worker function `_mp_worker` above, we see that it breaks the worker loop, adds the `_END` token to `out_q`, then exits. Next, on lines 9-12 we drain the `out_q` until all `_END` tokens have been retrieved. Lastly, on lines 15-16, we wait for the processes to exit.

#### Speed of Multiprocessing Dataloader

Using the same batch size of 300 as before, we can see how long it takes to loop over every batch:

```python
dataset = CustomDataset(n_samples=50000)
loader = MPBatchLoader(
    dataset,
    batch_size=300,
    num_workers=8,
    prefetch_batches=16,
)

now = time.time()
n_batches = 0
for batch in loader:
    n_batches += 1
    continue
print(f"Processed {n_batches} batches in {time.time() - now:.2f} seconds, time per batch {(time.time() - now)/n_batches:.4f} seconds")
```
```text
Processed 167 batches in 21.62 seconds, time per batch 0.1294 seconds
```

> So we see that the non-multiprocessing dataloader took 142 seconds to loop over every batch, whereas the multiprocesssing dataloader took only 22 seconds! Nearly 6.5 times faster! 

Whilst the example above gives an overview of the core ideas in the PyTorch DataLoader class, there are many additional features that are missing. The PyTorch class includes, but is not limited to, the following:
1. The batch indices are not pre-calculated as in our example, but made on the fly to allow for shuffling and specific sampling patterns. 
2. You can opt to drop the last batch if it has fewer samples in it than `batch_size`
3. Pinning of arrays in memory, allowing for faster transfers to the GPU.
4. Persistent workers, so the same workers are used over multiple epochs, rather than incurring the overhead of creating workers at the start of every epoch. 
5. Handling of dead/stuck workers. 



