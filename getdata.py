from googleapiclient.discovery import build
from IPython.display import JSON
import pandas as pd
import time
# How to get the Channel ID
# Go to the preferd Channel copy link
# insert the link hier https://commentpicker.com/youtube-channel-id.php
# put the Channel ID into the channels_IDs

api_key = 'AIzaSyAUr8JGDOQZoGSVU4EjHhI5M0j8jQhzDZ0'
channel_IDs = ['UCq2E1mIwUKMWzCA4liA_XGQ']

# Get credentials and create an API client
api_service_name = "youtube"
api_version = "v3"
youtube = build(
    api_service_name, api_version, developerKey=api_key)


def getChannelInfo(api_key,channel_IDs):
  channel_data = []
  request = youtube.channels().list(
    part="snippet,contentDetails,statistics",
    id=','.join(channel_IDs)
  )
  response = request.execute()

  # loop through items from the API responed and extract what you need
  for item in response['items']:
    data = {'channelName': item['snippet']['title'],
            'subcribers': item['statistics']['subscriberCount'],
            'views': item['statistics']['viewCount'],
            'videos': item['statistics']['videoCount'],
            'playlistID': item['contentDetails']['relatedPlaylists']['uploads']
    }

    channel_data.append(data)

  return pd.DataFrame(channel_data) 


def get_video_id(playlist_ID):
  video_ids = []

  request = youtube.playlistItems().list(
      part="snippet,contentDetails",
      playlistId=playlist_ID,
      maxResults = 100
  )
  response = request.execute()

  for item in response['items']:
      video_ids.append(item['contentDetails']['videoId'])

  next_page_token = response.get('nextPageToken')

  while next_page_token is not None:
      request = youtube.playlistItems().list(
                  part='contentDetails',
                  playlistId=playlist_ID,
                  maxResults = 50,
                  pageToken = next_page_token)
      response = request.execute()

      for item in response['items']:
          video_ids.append(item['contentDetails']['videoId'])

      next_page_token = response.get('nextPageToken')
      
  return video_ids


def getComments(videoIDs):
  comments = []
  counter = 0
  for videoID in videoIDs:
    if(counter % 20 == 0):
      print(round((counter * 100) / len(videoIDs)), '%')

    try:
      request = youtube.commentThreads().list(
        part='snippet,replies',
        videoId=videoID,
        maxResults=100
      )
      response = request.execute()

      videoComments = [comment["snippet"]['topLevelComment']['snippet']['textOriginal'] for comment in response['items'][0:100]]
      videoCommentsInfo = {'commentlen': len(videoComments),'videoID': videoID, 'comments': videoComments}

      comments.append(videoCommentsInfo)
  
    except:
      print('This video do not have comments or the comments could not get retrived ', videoID)

    counter += 1

  return pd.DataFrame(comments)


def main():
  start = time.time()
  
  channelInfo = getChannelInfo(api_key,channel_IDs)
  print(channelInfo)
  videoIDs = get_video_id(channelInfo["playlistID"][0])
  comments  = getComments(videoIDs)
  
  comments.to_csv('Comments.csv', encoding='utf-8')

  end = time.time()
  print("time:",end - start)

if __name__ == "__main__":
    main()
