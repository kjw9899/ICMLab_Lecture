import torch.utils.data as data
import numpy as np
from utils.utils import process_feat # feature를 원하는 크기만큼 segmentation
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import option
args=option.parse_args()

class Dataset(data.Dataset) :
    def __init__(self, args, transform=None, test_mode=False) :
        
        if self.dataset == 'UCF' :
            if test_mode :
                self.rgb_list_file = 'list/UCF-ResNeXt-test.list' # test에 사용될 npy(or plk) 파일의 dir를 text로 저장
            else :
                self.rgb_list_file = 'list/UCF-ResNeXt-train.list'
            
        elif self.dataset == 'XD' :
            if test_mode :
                self.rgb_list_file = 'list/XD-ResNeXt-test.list'
            else :
                self.rgb_list_file = 'list/XD-ResNeXt-train.list'
        
        elif self.dataset == 'TAD' :
            if test_mode :
                self.rgb_list_file = 'list/TAD-ResNeXt-test.list'
            else :
                self.rgb_list_file = 'list/TAD-ResNeXt-train.list'
        
        self.transform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None
        
    def _parse_lsit(self) :
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False :
            if args.datasetname == 'UCF' :
                print('UCF-Crime list')
                print(self.list)
            elif args.datasetname == 'XD' :
                print('XD-violence list')
                print(self.list)
            elif args.datasetname == 'TAD' :
                print('Traffic Accident Dataset list')
                print(self.list)
                
    
    def __getitem__(self, index) :
        if args.dataname == 'UCF' :
            features = np.load(self.list[index].strip('/n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1].strip('/n')[:-4]
        elif args.dataname == 'XD' :
            features = np.load(self.list[index].strip('/n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1].strip('/n')[:-4]
        elif args.dataname == 'TAD' :
            features = np.load(self.list[index].strip('/n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1].strip('/n')[:-4]
            
        if self.transform is not None :
            features = self.transform(features)
            
        if self.test_mode :
            return features
        
        else : # training
            # process 10-cropped clips feature
            # If I use feature magnitude, return divided_mag
            if args.daatasetname == 'UCF' :
                features = features.transpose(1, 0, 2) # [10, T= number of clips, Feature]
                divided_features = []
                
                divided_mag = []
                for feature in features :
                    feature = process_feat(feature, args.seg_length)
                    divided_features.append(feature)
                    divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
                divided_features = np.array(divided_features, dtype=np.float32)
                divided_mag = np.array(divided_mag, dtype=np.float32)
                
                return divided_features
            
            elif args.datasetname == 'XD':
                feature = process_feat(features, 10)
                if args.add_mag_info == True:
                    feature_mag = np.linalg.norm(feature, axis=1)[:, np.newaxis]
                    feature = np.concatenate((feature,feature_mag),axis = 1)
                return feature
            
            elif args.datasetname == 'TAD':
                feature = process_feat(features, 10)
                if args.add_mag_info == True:
                    feature_mag = np.linalg.norm(feature, axis=1)[:, np.newaxis]
                    feature = np.concatenate((feature,feature_mag),axis = 1)
                return feature
            
    def __len__(self) :
        return len(self.list)
    
    def get_num_frames(self) :
        return self.num_frame