# Face Recognition with OpenCV and Python
![interfaccia](https://github.com/ora-et-labora-team/awsu/assets/44711271/58c0ba65-6421-42ac-98a8-3f304346f98d)

AWSU is a simple GUI which shows how face recognizer have been done, using LBPH and DeepFace.

Dataset (gallery) is in `training-data` folder, subjects to probe is in `probe` folder.

LBPH             |  Deepface
:-------------------------:|:-------------------------:
![final](https://github.com/ora-et-labora-team/awsu/assets/44711271/993a6af5-799a-4e27-99a7-33e09558cafc)  |  ![final](https://github.com/ora-et-labora-team/awsu/assets/44711271/2feba32d-4131-43af-ba39-c5e6e2145dce)


## Requirements
- Python 3.9.11
- Linux or Windows (MacOS is untested)
- (dev) pre-commit

## Setup
1. Create a new virtualenvironment
```
python -m venv .env
```

2. Install requirements.txt
```
pip install -r requirements.txt
```

3. Install tkinter
```
sudo apt-get install python3-tk
```

4. Run the application
```
python main.py
```

5. (development) If you are a developer, install pre-commit by using `pip` and the `pre-commit` hook in the repo:
```
pip install pre-commit
pre-commit install
pre-commit run --all
``` 

## Usage
Click either on LBP or DeepFace button if you want to see how face recognizer is done using LBPH or DeepFace.

After having performed both of them, you can visualize the graphs to see what algorithm has a better performance.

In the images/LBP or images/Deepface there will be the faces that have been correctly detected by one of those algorithms.

In the images/graphs, you can see the graphs.

Our graphs are:
LBPH             |  Deepface
:-------------------------:|:-------------------------:
![LBPH](https://github.com/ora-et-labora-team/awsu/assets/44711271/36e0e2cd-7cd4-474b-88b0-36ab1bfc3c1f) |  ![DeepL](https://github.com/ora-et-labora-team/awsu/assets/44711271/48ad690c-cbe6-4180-8998-cd11d0e40ca8)


## Notes
- training data does not contains all the photos, rather than some of probed subjects
- LBPH is faster but scores worser than DeepFace (less than one minute vs. around three minutes on i7-11th)
