import flickrapi
from urllib.request import urlretrieve
import os
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv('FLICKER_API_KEY')
SECRET = os.getenv('FLICKER_API_SECRET')


def download_images(searchword, limitnum=1000):
    flickr = flickrapi.FlickrAPI(
        api_key=KEY, secret=SECRET, format='parsed-json')

    os.makedirs(f'./input_image/{searchword}', exist_ok=True)

    page = 1
    photos_downloaded = 0

    with tqdm(total=limitnum, desc=f"Downloading {searchword} images") as pbar:
        while photos_downloaded < limitnum:
            try:
                photos = flickr.photos.search(
                    text=searchword, extras='url_c', per_page=500, page=page, sort='relevance')

                if not photos['photos']['photo']:
                    print(f"No more photos found for {searchword}")
                    break

                for photo in photos['photos']['photo']:
                    if photos_downloaded >= limitnum:
                        break

                    try:
                        url = photo.get('url_c')
                        if url:
                            filename = f"./input_image/{searchword}/{searchword}-{photos_downloaded}.jpg"
                            urlretrieve(url, filename)
                            photos_downloaded += 1
                            pbar.update(1)
                    except Exception as e:
                        print(f"Error downloading image: {e}")

                page += 1

            except flickrapi.exceptions.FlickrError as e:
                print(f"Flickr API error: {e}")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                break

    print(f"Downloaded {photos_downloaded} images for {searchword}")


if __name__ == '__main__':
    with open('TW_Attractions_List.json', 'r', encoding='utf-8') as file:
        attractions = json.load(file)

    for attraction in attractions['TW_Attractions']:
        download_images(attraction)
