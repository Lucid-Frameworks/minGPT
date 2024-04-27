import torch


class DataLoader(torch.utils.data.DataLoader):
    """
    Separate the trainer and projects from torch DataLoader by creating a proxy.

    This DataLoader can be fed into tinygrad as well. We will eventually implement our own.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, 
        batch_sampler=None, num_workers=0, collate_fn=None,
        pin_memory=False, drop_last=False, timeout=0,
        worker_init_fn=None, *, prefetch_factor=2,
        persistent_workers=False):
        """ Torch Dataloader signature:

        DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
            batch_sampler=None, num_workers=0, collate_fn=None,
            pin_memory=False, drop_last=False, timeout=0,
            worker_init_fn=None, *, prefetch_factor=2,
            persistent_workers=False)

        https://pytorch.org/docs/stable/data.html
        """
        super().__init__(
            dataset=dataset,
            sampler=sampler,
            shuffle=shuffle,
            pin_memory=pin_memory,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
        ) # Not all params are passeed. We'll see.
