from selenium import webdriver 
from selenium.webdriver.common.keys import Keys

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
        
def PCCF(postal_codes):
    census_tracts = []
    driver = webdriver.Chrome()
    driver.get('http://www12.statcan.gc.ca/census-recensement/2011/dp-pd/prof/index.cfm?Lang=E')
    
    for code in postal_codes:
        button = driver.find_element_by_id("tabs1_2-lnk")
        button.click()
        element = driver.find_element_by_id("PCSearchText")
        element.send_keys(code + Keys.RETURN)
        census_tract_element = driver.find_element_by_id("wb-auto-8").text
        
        if (len(census_tract_element) > 0):
            census_tracts.append(find_between(census_tract_element, '\n', ','))
        else:
            census_tracts.append('nan')
        driver.get('http://www12.statcan.gc.ca/census-recensement/2011/dp-pd/prof/index.cfm?Lang=E')
    return census_tracts