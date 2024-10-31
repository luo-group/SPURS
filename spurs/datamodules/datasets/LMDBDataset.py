from torch.utils.data import  Dataset
import pickle 
import lmdb


class LMDBDataset(Dataset):
    def __init__(
        self,
        path,
        to_dict = True,
        to_list = True
) -> None:
        super().__init__()
        
        self.lmdb, self.lmdb_dict, self.lmdb_list = self.load_lmdb(path,
                                              to_dict=to_dict,
                                              to_list=to_list)
        
        
    def load_lmdb(
        self,
        lmdb_path,
        to_dict = True,
        to_list=True
):
        
        env = lmdb.open(lmdb_path,readonly=True)
        if to_dict:
            with env.begin() as txn:
                d = {key.decode():pickle.loads(value) for key, value in txn.cursor()}
        else:
            d = []
        if to_list:
            if to_dict:
                l = list(d.values())
            else:
                with env.begin() as txn:
                    l = [pickle.loads(value) for key, value in txn.cursor()]
        else:
            l = []
        return env, d, l
    
    def get_value(self,key):
        if len(self.lmdb_dict)>0:
            return self.lmdb_dict[key]
        else:
            with self.lmdb.begin() as txn:
                return pickle.loads(txn.get(key.encode()))
    
    
    def __len__(self):
        assert (self.lmdb_list)>0
        return len(self.lmdb_list)
    
    def __getitem__(self, idx):
        assert (self.lmdb_list)>0
        return self.lmdb_list[idx]
        