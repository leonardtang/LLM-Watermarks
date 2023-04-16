"""
Script to interact with (i.e. monitor) OpenAI's ChatGPT web interface.
"""

import pickle
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from undetected_chromedriver import Chrome


class ChatGPTBot:
    def __init__(self):
        self.driver = Chrome()
        self.driver.get("https://chat.openai.com/")
        self.cookies_path = "cookies.pkl"
        time.sleep(2)
        # Eventually have nice first-time cookie setup, like here: https://stackoverflow.com/questions/45417335/python-use-cookie-to-login-with-selenium
        try:
            cookies = pickle.load(open(self.cookies_path, "rb"))
        except:
            # First-time cookie setup. Login to ChatGPT then go to terminal to proceed
            input("Once cookies are saved, press enter...")
            cookies = self.driver.get_cookies()
            pickle.dump(cookies, open(self.cookies_path, "wb"))

        for cookie in cookies:
            try: 
                self.driver.add_cookie(cookie)
            except: 
                print("Warning: failed to set cookie", cookie)
                continue
        self.driver.get("https://chat.openai.com/")
        self.driver.implicitly_wait(0)
        time.sleep(2)


    def click_through_popups(self):
        """
        Click "Next" through initial explanatory popups
        """
        for xpath in ['//*[@id="headlessui-dialog-panel-:r1:"]/div[2]/div[4]/button', 
                      '//*[@id="headlessui-dialog-panel-:r1:"]/div[2]/div[4]/button[2]', 
                      '//*[@id="headlessui-dialog-panel-:r1:"]/div[2]/div[4]/button[2]']:
            button = self.driver.find_element(
                by=By.XPATH,
                value=xpath,
            )
            button.click()
            time.sleep(2)


    def send_response(self, text):
        # Again, sometimes the XPath changes
        while True:
            try:
                input_text_area = self.driver.find_element(By.XPATH, '//*[@id="__next"]/div[2]/div[2]/main/div[2]/form/div/div[2]/textarea')
            except:
                continue
            break
        input_text_area.send_keys(text)
        time.sleep(1)
        send_message = self.driver.find_element(By.XPATH, '//*[@id="__next"]/div[2]/div[2]/main/div[2]/form/div/div[2]/button')
        send_message.click()

    def get_response(self):
        # These are the loading ellipses that are present during text generation
        ellipses_one = '//*[@id="__next"]/div[2]/div[2]/main/div[2]/form/div/div[2]/button/div/span[2]'
        ellipses_two = '//*[@id="__next"]/div[2]/div[2]/main/div[2]/form/div/div[2]/button/div/span[3]'

        # Wait for elements to load
        WebDriverWait(self.driver, 20, poll_frequency=0.1).until(EC.presence_of_element_located((By.XPATH, ellipses_one)))
        WebDriverWait(self.driver, 20, poll_frequency=0.1).until(EC.presence_of_element_located((By.XPATH, ellipses_two)))

        # Wait for elements to disappear
        WebDriverWait(self.driver, 60, poll_frequency=0.1).until_not(EC.presence_of_element_located((By.XPATH, ellipses_one)))
        WebDriverWait(self.driver, 60, poll_frequency=0.1).until_not(EC.presence_of_element_located((By.XPATH, ellipses_two)))

        response_area = self.driver.find_element(By.XPATH, '//*[@id="__next"]/div[2]/div[2]/main/div[1]/div/div/div/div[2]/div/div[2]/div[1]/div/div/p')
        return response_area.text
    
    def query(self, text):
        self.send_response(text)
        return self.get_response()
    

if __name__ == "__main__":
    # Sometimes the popup XPaths change, so just retry until the right XPaths appear
    while True:
        try:
            print("Attempting to initialize...")
            bot = ChatGPTBot()
            bot.click_through_popups()
        except:
            print("Error during popup clickthrough")
            time.sleep(2)
            continue
        break
        
    query_text = "The quick brown fox"
    print(f"Querying with text {query_text}")
    for _ in range(10):
        response = bot.query(query_text)
        print("ChatGPT Response:")
        print(f"{response}\n")