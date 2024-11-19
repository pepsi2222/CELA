import os
import argparse
import sys
import time
import json
import random
import pandas as pd
import logging
from tqdm import tqdm
import multiprocessing

import requests
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=-1)
parser.add_argument("--links_file", type=str, default="tmdb.csv")
parser.add_argument("--output_file", type=str, default="tmdb.jsonl")
parser.add_argument("--multiprocess", type=int, default=1)
parser.add_argument("--existing_file", type=str)
args = parser.parse_args()


existing_movie_id = []
existing_info = []
if args.existing_file:
    with open(args.existing_file, 'r') as f:
        for line in f:
            content = json.loads(line)
            if len(content['movie_title']) > 0:
                existing_movie_id.append(content['movieID'])
                existing_info.append(content)
    
existing_movie_id = set(existing_movie_id)

SENTINEL = None


user_agents = [
    # Windows Chrome
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
    # macOS Safari
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
    # Linux Firefox
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
    # Android Chrome
    # 'Mozilla/5.0 (Linux; Android 11; SM-G960U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36',
    # iOS Safari
    # 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Mobile/15E148 Safari/604.1',
    # 添加其他 User-Agent
]


def extract_bdi_properties(soup):
    # 匹配 <strong><bdi>property</bdi></strong> 结构
    # 找到所有符合条件的<p>标签
    property_elements = soup.find_all('p', class_=None, recursive=True)

    # 初始化字典
    property_dict = {}

    # 遍历每个<p>标签，提取property和value，并存储到字典中
    for p_tag in property_elements:
        strong_tag = p_tag.find('strong')
        if strong_tag:
            bdi_tag = strong_tag.find('bdi')
            if bdi_tag:
                property_name = bdi_tag.text.strip()
                property_value = p_tag.contents[-1].strip() if p_tag.contents else ""
                property_dict[property_name] = property_value

    return property_dict


def scrap_with_tmdbID(movieID: int, tmdbID: int, sess=None, headers=None, type=str):
    # 发送HTTP请求获取网页内容
    time.sleep(random.random()*3)
    url = f"https://www.themoviedb.org/{type}/{tmdbID}"
    if sess is None:
        headers = {
            'User-Agent': user_agents[0],
            'Accept-Language': 'en-US,en;q=0.9',
        }
        response = requests.get(url, headers=headers)
    else:
        response = sess.get(url, headers=headers)

    movie_title = ""
    release_year = ""
    genres_list = []
    runtime = ""
    overview = ""
    people_dict = {}
    property_dict = {}
    keyword_list = []

    success = False

    # 检查请求是否成功
    if response.status_code == 200:
        success = True
        # 使用Beautiful Soup解析HTML内容
        soup = BeautifulSoup(response.text, 'html.parser')

        # 在这里可以编写代码来提取你需要的信息
        # 例如，如果你想要提取所有的链接，你可以使用：
        title_tag = soup.find('div', class_='title')
        if title_tag:
            _h_tag = title_tag.find('h2')
            if _h_tag:
                _a_tag = _h_tag.find('a')
            else:
                _a_tag = None
            movie_title = _a_tag.text.strip() if _a_tag else ""
        else:
            movie_title = ""

        release_year_ele = soup.find('span', class_='release')
        if release_year_ele:
            release_year = release_year_ele.text.strip()
        else:
            release_year = ""

        genres_span = soup.find('span', class_='genres')
        # 提取所有<a>标签中的文本
        if genres_span:
            genres_list = [genre.text.strip() for genre in genres_span.find_all('a')]
        else:
            genres_list = []

        run_time_span = soup.find('span', class_="runtime")
        if run_time_span:
            runtime = run_time_span.text.strip()
        else:
            runtime = ""

        overview_div = soup.find('div', class_="overview")
        if overview_div:
            _p_tag = overview_div.find('p')
            overview = _p_tag.text.strip() if _p_tag else ""
        else:
            overview = ""
        
        people_ol_ele = soup.find('ol', class_="people no_image")
        people_dict = {}
        if people_ol_ele:
            profiles = people_ol_ele.find_all('li', class_='profile')
            if profiles:
                for profile in profiles:
                    name = profile.find('p').find('a').text.strip()
                    role = profile.find('p', class_='character').text.strip()
                    people_dict[role] = name

        property_col = soup.find('section', class_="split_column")
        if property_col:
            property_dict = extract_bdi_properties(property_col)
        else:
            property_dict = {}

        keyword_sec = soup.find('section', class_="keywords")
        keyword_list = []
        if keyword_sec:
            _a_list = keyword_sec.find_all('a')
            if _a_list:
                keyword_list = [_a.text.strip() for _a in _a_list]
        
    content = {
        "movieID": int(movieID),
        "tmdID": int(tmdbID),
        "url": url,
        "movie_title": movie_title,
        "release_year": release_year,
        "genre_list": genres_list,
        "runtime": runtime,
        "overview": overview,
        "people_dict": people_dict,
        "property_dict": property_dict,
        "keyword_list": keyword_list
    }

    return success, content

def scrap_with_list(df, valid_index, s, e, queue):
    sess = requests.Session()
    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept-Language': 'en-US,en;q=0.9',
        # 添加其他请求头信息
    }
    tbar = tqdm(total=e-s, ncols=100)
    n_success = 0
    contents = []
    # for row in df[s:e].itertuples(index=False):
    n = 0
    for i in valid_index[s:e]:
        row = df.iloc[i]
        movie_id = row.movieId
        tmdb_id = int(row.tmdbId)
        type = row.type
        if movie_id not in existing_movie_id:
            success, content = scrap_with_tmdbID(movie_id, tmdb_id, sess=None, headers=headers, type=type)
            n_success += success
            if success:
                contents.append(content)
            else:
                time.sleep(random.random()*3)
                sess = requests.Session()
                headers['User-Agent'] = random.choice(user_agents)
        else:
            n_success += 1
        tbar.update(1)
        n += 1
    
    queue.put((n_success, contents))
    queue.put(SENTINEL)
    sess.close()
    print("End.")
    return


# Configure logging
logging.basicConfig(
    filename='spider.log',  # Specify the name of the log file
    level=logging.INFO,  # Set the logging level (INFO, WARNING, ERROR, etc.)
    format='%(asctime)s [%(levelname)s] %(message)s',  # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S'  # Define the date and time format
)


movie_id_df = pd.read_csv(args.links_file, header=0)[['movieId', 'tmdbId', 'type']]
movie_id_df = movie_id_df.fillna(-1)
# movie_id_df = movie_id_df.astype(int)

start_index = args.start#int(sys.argv[1])
end_index = len(movie_id_df) if args.end<0 else args.end#int(sys.argv[2])

if end_index <= start_index:
    raise ValueError(f"`end` must be larger than `start`, while got {end_index} <= {start_index}.")

result_filename = args.output_file

n_success = 0

processes = []
if args.multiprocess > 1:   # 多进程
    n_proc = args.multiprocess
    movie_id_array = movie_id_df['movieId'].tolist()
    valid_index = [i for i in range(start_index, end_index) if movie_id_array[i] not in existing_movie_id]
    # num_per_process = (end_index - start_index) // n_proc
    # remainder = (end_index - start_index) % n_proc
    num_per_process = len(valid_index) // n_proc
    remainder = len(valid_index) % n_proc
    index_list = []
    s = 0
    for i in range(n_proc):
        if i < remainder:
            e = s + num_per_process + 1
        else:
            e = s + num_per_process
        index_list.append((s, e))
        s = e

    result_queue = multiprocessing.Queue()
    for i in range(n_proc):
        process = multiprocessing.Process(target=scrap_with_list, args=(movie_id_df, valid_index, index_list[i][0], index_list[i][1], result_queue))
        process.start()
        processes.append(process)

    n_success = len(existing_movie_id)
    seen_sentinel_count = 0
    while seen_sentinel_count < len(processes):
        result = result_queue.get()
        if result is SENTINEL:
            seen_sentinel_count += 1
            print(f"Here: {seen_sentinel_count}.")
        else:
            n_success += result[0]
            with open(result_filename, 'a') as f:
                for result in result[1]:
                    try:
                        json.dump(result, f)
                        f.write("\n")
                    except:
                        print(result)

    for process in processes:
        process.join()
    
else:
    movie_id_array = movie_id_df['movieId'].tolist()
    valid_index = [i for i in range(start_index, end_index) if movie_id_array[i] not in existing_movie_id]
    tbar = tqdm(total=len(valid_index), ncols=100)

    n_success = len(existing_movie_id)
    for i in valid_index:
        row = movie_id_df.iloc[i]
        movie_id = row.movieId
        tmdb_id = int(row.tmdbId)
        type = row.type

        if tmdb_id > 0:
            try:
                success, content = scrap_with_tmdbID(movieID=movie_id, tmdbID=tmdb_id, type=type)
                if success:
                    n_success += 1

            except KeyboardInterrupt as e:
                logging.info("Keyboard interrupt.")
                break
            except Exception as e:
                logging.error(f"Error occured when processing {tmdb_id}. Error: {e}")
                content = {"movieID": movie_id, "tmdbID": tmdb_id}

        if tbar:
            tbar.update(1)

        # write file
        if success:
            with open(result_filename, 'a') as f:
                json.dump(content, f)
                f.write("\n")

logging.info(f"Finished {n_success} / {(end_index-start_index)}. Saved in {args.output_file}.")

logging.shutdown()
