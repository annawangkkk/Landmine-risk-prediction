from PIL import Image
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from ..utils import *

class Landmine(Dataset):
    def __init__(self, root, test_name, split, map_type, transform=None):
        imputer = KNNImputer(n_neighbors = 10, weights = 'distance')
        scaler = StandardScaler()
        self.root = root
        self.transform = transform
        self.map_type = map_type
        self.image_root = f'/home/siqiz/social_good/data/{self.map_type}'
        self.split = split
        self.test_name = test_name # three splits
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cols_info = FeatureNames('new')
        numeric_cols = cols_info.numeric_cols
        binary_cols = cols_info.binary_cols
        features = numeric_cols + binary_cols
        if self.split == "train_labeled":
            self.df = pd.read_csv(self.root + f'/processed_dataset/{test_name}/train/train_labeled.csv',index_col=0)
            extra_info = np.array(self.df[['mines_outcome','LATITUD_Y' ,'LONGITUD_X']])
            self.df = self.df[features]
            self.df = imputer.fit_transform(self.df)
            self.df = scaler.fit_transform(self.df)
            self.df = np.concatenate([self.df,extra_info],axis=1)
        elif self.split == "test_labeled":
            self.df = pd.read_csv(self.root + f'/processed_dataset/{test_name}/test/test_labeled.csv',index_col=0)
            extra_info = np.array(self.df[['mines_outcome','LATITUD_Y' ,'LONGITUD_X']])
            train_df = pd.read_csv(self.root + f'/processed_dataset/{test_name}/train/train_labeled.csv',index_col=0)
            train_df = train_df[features]
            self.df = self.df[features]
            imputer.fit(train_df)
            self.df = imputer.transform(self.df)
            scaler.fit(train_df)
            self.df = scaler.transform(self.df)
            self.df = np.concatenate([self.df,extra_info],axis=1)
            del train_df
        self.df = pd.DataFrame(self.df, columns=features + ['mines_outcome','LATITUD_Y' ,'LONGITUD_X'])
        self.img_paths = np.array(self.image_root + '/' + self.df['LATITUD_Y'].astype(str) + '_' + self.df['LONGITUD_X'].astype(str) + '.jpg')
        self.targets = np.array(self.df['mines_outcome'])
        self.df = torch.from_numpy(np.array(self.df[features])).to(device)
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self,idx):
        img_name = self.img_paths[idx]
        img = Image.open(self.img_paths[idx]).crop((0,0,224,224)) # remove "Google"
        tar = self.targets[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        tab = self.df[idx]
        return img, tab, torch.tensor(tar).long()