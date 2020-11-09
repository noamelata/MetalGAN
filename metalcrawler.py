import aiohttp
import os, sys
from PIL import Image
from bs4 import BeautifulSoup
from io import BytesIO
import numpy as np
import pandas as pd
import asyncio
from asgiref.sync import *
from itertools import chain
import tqdm


BASE_URL = "https://www.metal-archives.com/"
DATA_PATH = r""
SEED = [("Insomnium", r"https://www.metal-archives.com/bands/Insomnium/2332"),
        ]


def concatenate(*lists):
    return chain(*lists)

async def fetch(session, url):
    async with session.get(url) as response:
        assert response.status == 200
        return await response.read()


def make_square(im, min_size=256, fill_color=(0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('L', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


async def get_logo(soup, session):
    search = soup.find(id="logo")
    assert search
    logo_url = search['href']
    imgbytes = await fetch(session, logo_url)
    img = Image.open(BytesIO(imgbytes))

    img = make_square(img).resize(size=(512, 512))
    return img


async def get_similar_artists(soup, session, threshhold=5):
    table_url = soup.find(title="Similar artists")['href']

    sp = BeautifulSoup(await fetch(session, table_url), features="lxml")

    tb = sp.find('table')
    df = pd.read_html(str(tb), encoding='utf-8', header=0)[0]
    df['href'] = [np.where(tag.has_attr('href'), tag.get('href'), "no link") for tag in tb.find_all('a')]
    df = df[:-1]
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    df = df[df['Score'] >= threshhold]
    return df


async def parse_band(name, url, session):
    soup = BeautifulSoup(await fetch(session, url), features="lxml")
    img = await get_logo(soup, session)

    try:
        os.mkdir(os.path.join(DATA_PATH, name))
    except OSError:
        pass
    try:
        img.save(os.path.join(DATA_PATH, name, name+".png"))
    except OSError:
        pass

    similar_artists = await get_similar_artists(soup, session)
    return similar_artists


async def branch(name, url, searched, lock, session, t):
    try:
        similar_artists = await parse_band(name, url, session)
    except (AssertionError, aiohttp.ServerDisconnectedError, aiohttp.ClientOSError, TimeoutError):
        try:
            similar_artists = await parse_band(name, url, session)
        except Exception:
            return []
    except (TypeError, OSError):
        return []
    except Exception as e:
        print(e)
        return []

    #print(name)


    seed = []
    await lock.acquire()
    #print(name, "has lock")
    try:
        for indx, row in similar_artists.iterrows():
            new_name, new_url = row['Name'], str(row['href'])
            if new_name not in searched:
                seed.append((new_name, new_url))
                searched.append(new_name)
    except TypeError as e:
        print(e)
    finally:
        t.update()
        lock.release()
        #print(name, "released lock")

    await asyncio.sleep(0.01)
    return seed


async def main(seed=SEED):
    lock = asyncio.Lock()
    max_depth = 12
    max_bands = 10000
    depth = 0
    searched = [name for name, url in seed]
    timeout = aiohttp.ClientTimeout(total=60*60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while (len(seed) > 0 and len(searched) < max_bands and depth < max_depth):
            desc = "run number " + str(depth)
            t = tqdm.tqdm(list(range(len(seed))), desc=desc)
            tasks = []
            new_seed = []
            for name, url in seed:
                task = asyncio.create_task(branch(name, url, searched, lock, session, t))
                tasks.append(task)
            results = await asyncio.gather(*tasks, return_exceptions=False)
            new_seed = list(concatenate(*results))
            seed = new_seed.copy()
            depth += 1
            t.close()
            await asyncio.sleep(2)
    print(len(searched))
    return 0

if __name__ == "__main__":
    asyncio.run(main())



