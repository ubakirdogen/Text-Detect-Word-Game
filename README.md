# Text Detection from a TV Word Game Show

## 1. Description
This small project aims to detect question/answer texts from videos of a popular word game show which is hosted on a Turkish TV channel. To have a better understanding of the game, check the sample video in .\video\sample_vid.mp4
After procesing the video, the script outputs all the question/answer pairs in a text file into the .\output folder as well as the screenshots of the frames where the answer is revealed.
More videos can be downloaded for processing from [official website of the show](https://www.teve2.com.tr/programlar/guncel/kelime-oyunu/bolumler) using a video downloader chrome extension.
This project uses OpenCV and Tesseract OCR Engine. Tested on Python 3.10

## 2. Run on local

### 2.1 Requirements
```bash
$ pip install -r requirements.txt
```

### 2.2. Usage
```bash
$ python detect_qns_ans.py [input_video_path] [start_marker_in_secs -optional] [end_marker_in_secs -optional]
```

### 2.3. Demo
```bash
$ python detect_qns_ans.py .\video\sample_vid.mp4 
```


