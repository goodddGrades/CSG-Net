import torch
from parser import parameter_parser
from main import main
from utils import set_seed
import save_results
main = save_results.wrap(main)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible")
    warnings.filterwarnings("ignore")
    args = parameter_parser()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    model_types = ['BiSGTAR']
    args.seed = 1
    results={}

    args.data = 4
    args.similarity_method = 'gaussian'
    args.alpha = 0.5
    args.beta = 0.4
    args.gama = 0.05
    args.use_contrastive = True
    args.cl_proj_dim = 64
    args.cl_weight = 0.1
    args.use_xgboost = True
    args.model_type = 'BiSGTAR'
    print('Now model is: ', args.model_type, ' use_xgboost:', args.use_xgboost)
    set_seed(args.seed)
    main(args)
