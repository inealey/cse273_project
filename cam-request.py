#!/usr/bin/python3
import threading
from time import localtime, strftime
from pathlib import Path
import requests
from requests.auth import HTTPBasicAuth

#OUTPUT_PATH = '/data/tower_images/'
OUTPUT_PATH = '/Users/inealey/Documents/calit2/jg-tower/tower-image-request/out/'
#SECRET_PATH = '/etc/tower-auth/'
SECRET_PATH = '/Users/inealey/Documents/calit2/jg-tower/tower-image-request/sec/'

with open(SECRET_PATH + 'user') as f:
    USER = f.read().strip('\n')
    f.close()
with open(SECRET_PATH + 'password') as f:
    PASS = f.read().strip('\n')
    f.close()

class imgThread (threading.Thread):
   def __init__(self, threadID, name, url):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.url = url
   def run(self):
      print ("Starting thread: " + self.name)
      dl_img(self.name, OUTPUT_PATH, self.url)


def dl_img(threadName, imgPath, cameraUrl):
   r = requests.get(cameraUrl, auth=HTTPBasicAuth(USER, PASS))
   ts = localtime()
   timeOfDay = strftime("%Y_%m_%d/%H/", ts)
   imgName = threadName + '_' + strftime("%a_%d_%Y_%H-%M-%S", ts)
   Path(imgPath + timeOfDay + threadName).mkdir(parents=True, exist_ok=True)
   with open(imgPath + timeOfDay + threadName + '/' + imgName + '.jpg', 'wb') as outfile:
      outfile.write(r.content)

def initUrlMap():
   d={}

   d['north'] = 'http://72.0.39.175/cgi-bin/image.jpg?textdisplay=disable&size=3072x2048&quality=100'
   d['south'] = 'http://72.0.39.176/cgi-bin/image.jpg?textdisplay=disable&size=3072x2048&quality=100'
   d['east'] = 'http://72.0.39.177/cgi-bin/image.jpg?textdisplay=disable&size=3072x2048&quality=100'
   d['west'] = 'http://72.0.39.178/cgi-bin/image.jpg?textdisplay=disable&size=3072x2048&quality=100'

   return d

def initThreads(threadList, urlDict):
   thread0 = imgThread(0, 'North', urlDict['north'])
   thread1 = imgThread(1, 'East', urlDict['east'])
   thread2 = imgThread(2, 'South', urlDict['south'])
   thread3 = imgThread(3, 'West', urlDict['west'])

   threadList.append(thread0)
   threadList.append(thread1)
   threadList.append(thread2)
   threadList.append(thread3)

   for t in threadList:
     t.start()

if __name__ == '__main__':
   threads = []
   camDict=initUrlMap()
   initThreads(threads, camDict)

   # Wait for all threads to complete
   for t in threads:
      t.join()
   print ("Exiting")
