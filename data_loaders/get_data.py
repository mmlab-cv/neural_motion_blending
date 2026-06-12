from torch.utils.data import DataLoader
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import Truebones

def get_dataset_class(name):
    return Truebones

def get_dataset(num_frames, split='train', temporal_window=31, t5_name='t5-base', balanced=False, objects_subset="all", dataset="truebones"):
    dataset = Truebones(split=split, num_frames=num_frames, temporal_window=temporal_window, t5_name=t5_name, balanced=balanced, objects_subset=objects_subset, dataset=dataset)
    return dataset


def get_dataset_loader(batch_size, num_frames, split='train', temporal_window=31, t5_name='t5-base', balanced=True, objects_subset="all", dataset="truebones"):
    dataset = get_dataset(num_frames=num_frames, split=split, temporal_window=temporal_window, t5_name=t5_name, balanced=balanced, objects_subset=objects_subset, dataset=dataset)
    collate = truebones_batch_collate
    sampler = None
    if balanced: #create batch sampler
        from data_loaders.truebones.data.dataset import TruebonesSampler
        sampler = TruebonesSampler(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=True if sampler is None else False,
        num_workers=8, drop_last=True, collate_fn=collate
    )
    return loader