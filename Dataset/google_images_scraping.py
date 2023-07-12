import os
import selenium
from selenium import webdriver
import base64
import time
import urllib.request
from selenium.webdriver.common.by import By

"""
SCRAPED DATA LINKS FOR REPLICATION:

------ apple -------
https://www.google.com/search?client=ubuntu&hs=2Cm&channel=fs&q=sliced+apple&tbm=isch&sa=X&ved=2ahUKEwjdgoSQ2NL_AhVvFVkFHYM4ARoQ0pQJegQIDBAB&biw=2488&bih=1328&dpr=1
https://www.google.com/search?q=cut+apple&tbm=isch&ved=2ahUKEwjRv7WTntL_AhWqAWIAHd0GBokQ2-cCegQIABAA&oq=cut+apple&gs_lcp=CgNpbWcQAzIHCAAQigUQQzIHCAAQigUQQzIHCAAQigUQQzIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQ6BggAEAcQHlDQBVjNCGCLDGgAcAB4AIABZYgB1wKSAQMzLjGYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=Xs-RZNGYBaqDiLMP3Y2YyAg&bih=1328&biw=2488&client=ubuntu&hs=Jp2
https://www.google.com/search?q=apple+half&tbm=isch&ved=2ahUKEwiX24nR2NL_AhV3J2IAHX5EBEYQ2-cCegQIABAA&oq=apple+half&gs_lcp=CgNpbWcQAzIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQ6BwgAEIoFEEM6CggAEIoFELEDEEM6CAgAEIAEELEDOgsIABCABBCxAxCDAVDDEFiSGmCLG2gAcAB4AIABtwGIAasHkgEEMTAuMZgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=sAySZJfDJ_fOiLMP_oiRsAQ&bih=1328&biw=2488&client=ubuntu&hs=2Cm
https://www.google.com/search?q=apple+fruit+cut+pieces&tbm=isch&ved=2ahUKEwjY1MW92dL_AhVAE1kFHf5ZDmEQ2-cCegQIABAA&oq=apple+fruit+cut+pieces&gs_lcp=CgNpbWcQAzoHCAAQigUQQzoFCAAQgAQ6BggAEAgQHjoECAAQHlCxA1jtDmC-D2gAcAB4AIABYIgB3QSSAQE4mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=lA2SZNi6B8Cm5NoP_rO5iAY&bih=1328&biw=2488&client=ubuntu&hs=2Cm

------ apricot -----



------- banana ------
https://www.google.com/search?client=ubuntu&hs=Wwm&channel=fs&q=cut+banana&tbm=isch&sa=X&ved=2ahUKEwiYwMC42tn_AhXXF1kFHVkODsoQ0pQJegQICRAB&biw=2488&bih=1328&dpr=1
https://www.google.com/search?q=sliced+banana&tbm=isch&ved=2ahUKEwiAiee52tn_AhXZJGIAHUP2AwMQ2-cCegQIABAA&oq=sliced+banana&gs_lcp=CgNpbWcQAzIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeOgcIABCKBRBDOgUIABCABFCyB1jODGD-DGgAcAB4AIABYYgBgASSAQE3mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=HrqVZICCNtnJiLMPw-yPGA&bih=1328&biw=2488&client=ubuntu&hs=Wwm
https://www.google.com/search?q=banana+pieces&tbm=isch&ved=2ahUKEwjivNfF2tn_AhWlBVkFHSOCBhgQ2-cCegQIABAA&oq=banana+pieces&gs_lcp=CgNpbWcQAzIFCAAQgAQyBQgAEIAEMgUIABCABDIGCAAQBRAeMgYIABAIEB4yBggAEAgQHjIGCAAQCBAeMgYIABAIEB4yBggAEAgQHjIGCAAQCBAeOgcIABCKBRBDOgoIABCKBRCxAxBDOggIABCABBCxA1CwF1i8JGCuJWgAcAB4AYABkAGIAZAIkgEEMTMuMZgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=N7qVZKLFMKWL5NoPo4SawAE&bih=1328&biw=2488&client=ubuntu&hs=Wwm



----- avocado ------


----- banana -------


------ peach -------


----- orange ------


----- strawberry -----


----- pear -----


----- kiwi ------


------ lemon -------


----- pineapple ------


---- cucumber ------
https://www.google.com/search?client=ubuntu&hs=k1R&channel=fs&q=cucumber+slices&tbm=isch&sa=X&ved=2ahUKEwjcgcWH39L_AhVKE1kFHXchAcoQ0pQJegQIDBAB&biw=2488&bih=1328&dpr=1
https://www.google.com/search?q=cut+cucumber&tbm=isch&ved=2ahUKEwijk9CI39L_AhVCFmIAHfBeBecQ2-cCegQIABAA&oq=cut+cu&gs_lcp=CgNpbWcQARgAMgcIABCKBRBDMgUIABCABDIFCAAQgAQyBwgAEIoFEEMyBQgAEIAEMgcIABCKBRBDMgUIABCABDIHCAAQigUQQzIHCAAQigUQQzIFCAAQgAQ6CAgAEIAEELEDUI8SWN4XYN4haABwAHgAgAFhiAG5BJIBATeYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=bxOSZOOfLsKsiLMP8L2VuA4&bih=1328&biw=2488&client=ubuntu&hs=k1R
https://www.google.com/search?q=cut+cucumber+long&tbm=isch&ved=2ahUKEwiwk-Db39L_AhWeF2IAHeFrCqwQ2-cCegQIABAA&oq=cut+cucumber+long&gs_lcp=CgNpbWcQAzoHCAAQigUQQzoFCAAQgAQ6BggAEAgQHjoHCAAQGBCABFC6BVjBEWDVEmgAcAB4AIABuAKIAfwHkgEHMC4xLjIuMZgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=HhSSZLCMBZ6viLMP4dep4Ao&bih=1328&biw=2488&client=ubuntu&hs=k1R

---- tomato ------



------ bell pepper ------



---- carrot -----


----- cabbage -----



---- cauliflower ----


---- brocoli -----


---- eggplant ----


----- lettuce ------


----- corn ------



------ onion -------



----- lime ------



------ orange -----



------ pineapple ------




"""

SAVE_FOLDER = 'scraped_data'
FRUIT_CLASS = 'carrot'
GOOGLE_IMAGES = 'https://www.google.com/search?q=slice+carrot&tbm=isch&ved=2ahUKEwj-s-vy7ISAAxUlBGIAHcTFCj4Q2-cCegQIABAA&oq=slice+carrot&gs_lcp=CgNpbWcQAzIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeOgcIABCKBRBDOgUIABCABDoKCAAQigUQsQMQQzoICAAQgAQQsQNQwgZYwRdg_xdoAnAAeACAAXKIAaMFkgEDOC4xmAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=01isZL7-BKWIiLMPxIur8AM&bih=1328&biw=2488&client=ubuntu&hs=c3K'
driver = webdriver.Firefox()
driver.get(GOOGLE_IMAGES)

def scroll_to_end():
    """
    Functionality to ensure we are scraping the entire first google image results page.
    """
    driver.execute_script("window.scrollTo(0, document.body.scrollHeights);")
    time.sleep(5)
    print('done scrolling...')

    # TODO: modify scroll script to go farther

def scroll_n_times(n):
    for i in range(n):
        # driver.execute_script("window.scrollTo(0, document.body.scrollHeights);")
        driver.execute_script("window.scrollTo(0, 10000);")
        # wait to load page
        time.sleep(0.5)


counter = 509
for i in range(1,2):
    # scroll_to_end()
    scroll_n_times(5)

    image_elements = driver.find_elements(By.CSS_SELECTOR, 'img.rg_i') # 'img.rg_i.Q4LuWd'
    print("Image Elements: ", len(image_elements))

    for image in image_elements:
        if (image.get_attribute('src') is not None):
            my_image = image.get_attribute('src').split('data:image/jpeg;base64,')
            filename = SAVE_FOLDER + '/' + FRUIT_CLASS + '/' + FRUIT_CLASS + str(counter) + '.jpeg'
            if (len(my_image) > 1):
                with open(filename, 'wb') as f:
                    f.write(base64.b64decode(my_image[1]))
            else:
                print("Image Attribute: ", image.get_attribute('src'))
                urllib.request.urlretrieve(image.get_attribute('src'), filename)
            counter+=1