# Preparation
## set up python environment 
We provide the environment with the amazing tool [pixi](https://pixi.sh/latest/). 
You can set up the python environment as simple as running `pixi install`

## download EuRoC dataset 
`python scripts/get_euroc.py --target_dir "euroc_dataset"`

# Training and Evaluation 
- train: 

    `python model.py --mode train --name my_experiment --dataset euroc --rate_mode imu_rate` 
- evaluation

    `python model.py --mode infer --name my_experiment --dataset euroc --rate_mode imu_rate`


# Acknowledgement
We thank [AirIMU](https://github.com/haleqiu/AirIMU) and [HuggingFace diffusers](https://github.com/huggingface/diffusers) for making their codebase open-source. Our code is built on top of them, for loading EuRoC dataset and building diffusion model network respectively. 