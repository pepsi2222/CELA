import os
import sys
import time
import json
import random
import pandas as pd
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging


username = 'pepsiness'
password = 'dygjej-jiszy6-Hotbax'

start_index = int(sys.argv[1])
end_index = int(sys.argv[2])

# Configure logging
logging.basicConfig(
    filename='spider.log',  # Specify the name of the log file
    level=logging.INFO,  # Set the logging level (INFO, WARNING, ERROR, etc.)
    format='%(asctime)s [%(levelname)s] %(message)s',  # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S'  # Define the date and time format
)

opt = ChromeOptions()
opt.add_argument("--headless")

driver = Chrome(options=opt)
driver.get("https://movielens.org/login")

try:
    time.sleep(random.randint(3, 5))
    # Wait for the username input field to be present
    username_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//input[@formcontrolname="userName"]'))
    )
    
    # Enter the username
    username_input.send_keys(username)

    time.sleep(random.randint(3, 5))
    # Locate the password input field
    password_input = driver.find_element(By.XPATH, '//input[@formcontrolname="password"]')
    
    # Enter the password
    password_input.send_keys(password)

    time.sleep(random.randint(3, 5))
    # Submit the form
    submit_button = driver.find_element(By.XPATH, '//button[@type="submit"]')
    submit_button.click()


except Exception as e:
    logging.exception('An exception occurred: %s', e)
    exit()

m_ids = pd.read_csv('cant_remap_tmdb.csv', header=0)['movieId']
d = dict()
for id in m_ids[start_index: end_index]:
    movie = {
        'movieID': id,
        'url': f'https://movielens.org/movies/{id}'
    }
    try:
        time.sleep(random.randint(3, 5))
        driver.get(movie['url'])

        time.sleep(random.randint(3, 6))
        movie['movie_title'] = driver.find_element(By.CLASS_NAME, "movie-title").text.strip()
        try:
            movie['overview'] = driver.find_element(By.CLASS_NAME, "lead.plot-summary").text
        except:
            movie['overview'] = ""
        try:
            tags = driver.find_elements(By.CLASS_NAME, "ml4-tag-main-btn")
            movie['keyword_list'] = [_.text for _ in tags]
        except:
            movie['keyword_list'] = []
        try:
            movie['average_rating'] = driver.find_element(By.XPATH, '//div[contains(@class, "movie-details-heading") and contains(text(), "Average of")]/following-sibling::div').text.replace('stars', '').strip()
        except:
             movie['average_rating'] = ""
        try:
            movie['property_dict'] = {
                'original_language': driver.find_element(By.XPATH, '//div[@class="movie-details-heading" and text()="Languages"]/following-sibling::span/a').text.strip(),
            }
        except:
            movie['property_dict'] = {}

            attrs = driver.find_elements(By.XPATH, '//ul[@class="movie-attr"]/li')
        try:
            movie['release_year'] = attrs[0].text.strip()
        except:
            movie['release_year'] = ""
        try:
            movie['runtime'] = attrs[-1].text.strip()
        except:
            movie['runtime'] = ""
        try:
            genres = driver.find_elements(By.CSS_SELECTOR, 'a[uisref="exploreGenreShortcut"]')
            movie["genre_list"] = [_.text.strip() for _ in genres]
        except:
            movie["genre_list"] = []
        people_dict = {}
        directors = driver.find_elements(By.XPATH, '//div[@class="movie-details-heading" and text()="Directors"]/following-sibling::span/a')
        if directors:
            people_dict["Director"] = [_.text.strip() for _ in directors]
        movie['people_dict'] = people_dict

        with open('movielens.json', 'a') as f:
            json.dump(movie, f)
            f.write('\n')
        

    except Exception as e:
        logging.exception(f'{id} | An exception occurred: %s', e)

# Close the browser window
driver.quit()

logging.shutdown()

