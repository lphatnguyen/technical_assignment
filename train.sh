# Running training for FER+ dataset
python src/train.py --base_folder ferPlus2016/data --ferplus_path ferPlus2016/fer2013/fer2013new.csv --epochs 50 --saving_fn best_weights_lenet_multi --model lenet --is_ferplus 1 --batch_size 512
python src/train.py --base_folder ferPlus2016/data --ferplus_path ferPlus2016/fer2013/fer2013new.csv --epochs 50 --saving_fn best_weights_resnet18_multi --model resnet18 --is_ferplus 1 --batch_size 512
python src/train.py --base_folder ferPlus2016/data --ferplus_path ferPlus2016/fer2013/fer2013new.csv --epochs 50 --saving_fn best_weights_resnet34_multi --model resnet34 --is_ferplus 1 --batch_size 512
python src/train.py --base_folder ferPlus2016/data --ferplus_path ferPlus2016/fer2013/fer2013new.csv --epochs 50 --saving_fn best_weights_mobilenet_multi --model mobilenet_v2 --is_ferplus 1 --batch_size 512

# Running training for FER dataset
python src/train.py --base_folder ferPlus2016/data --ferplus_path ferPlus2016/fer2013/fer2013new.csv --epochs 50 --saving_fn best_weights_lenet_single --model lenet --is_ferplus 0 --batch_size 512
python src/train.py --base_folder ferPlus2016/data --ferplus_path ferPlus2016/fer2013/fer2013new.csv --epochs 50 --saving_fn best_weights_resnet18_single --model resnet18 --is_ferplus 0 --batch_size 512
python src/train.py --base_folder ferPlus2016/data --ferplus_path ferPlus2016/fer2013/fer2013new.csv --epochs 50 --saving_fn best_weights_resnet34_single --model resnet34 --is_ferplus 0 --batch_size 512
python src/train.py --base_folder ferPlus2016/data --ferplus_path ferPlus2016/fer2013/fer2013new.csv --epochs 50 --saving_fn best_weights_mobilenet_single --model mobilenet_v2 --is_ferplus 0 --batch_size 512
