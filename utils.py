import platform
import os
import time
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium import webdriver
import time
from webdriver_manager.chrome import ChromeDriverManager
import logging

def create_driver_selenium():
    options = get_chrome_browser_options()  # Use the method to get Chrome options

    chrome_install = ChromeDriverManager().install()
    folder = os.path.dirname(chrome_install)
    if platform.system() == "Windows":
        chromedriver_path = os.path.join(folder, "chromedriver.exe")
    else:
        chromedriver_path = os.path.join(folder, "chromedriver")
    service = ChromeService(executable_path=chromedriver_path)
    return webdriver.Chrome(service=service, options=options)

def HTML_to_PDF(FilePath):
    if not os.path.isfile(FilePath):
        raise FileNotFoundError(f"The specified file does not exist: {FilePath}")
    
    # Convert to absolute path and proper URL format
    FilePath = f"file:///{os.path.abspath(FilePath).replace(os.sep, '/')}"
    driver = None
    
    try:
        # Initialize driver with retry mechanism
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                driver = create_driver_selenium()
                driver.set_page_load_timeout(180)
                driver.get(FilePath)
                break
            except Exception as e:
                retry_count += 1
                logging.warning(f"Attempt {retry_count} failed: {str(e)}")
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass
                if retry_count == max_retries:
                    raise RuntimeError(f"Failed to initialize Chrome driver after {max_retries} attempts")
                time.sleep(2)  # Wait before retrying
        
        # Wait for the page to be fully loaded
        time.sleep(2)
        
        # Wait for the body element to be present
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Additional wait to ensure all resources are loaded
        time.sleep(1)
        
        # Generate PDF with retry mechanism
        retry_count = 0
        while retry_count < max_retries:
            try:
                pdf_base64 = driver.execute_cdp_cmd("Page.printToPDF", {
                    "printBackground": True,
                    "landscape": False,
                    "paperWidth": 8.27,
                    "paperHeight": 11.69,
                    "marginTop": 0.8,
                    "marginBottom": 0.8,
                    "marginLeft": 0.5,
                    "marginRight": 0.5,
                    "displayHeaderFooter": False,
                    "preferCSSPageSize": True,
                    "generateDocumentOutline": False,
                    "generateTaggedPDF": False,
                    "transferMode": "ReturnAsBase64"
                })
                
                if not pdf_base64 or 'data' not in pdf_base64:
                    raise RuntimeError("No data received from Chrome")
                    
                return pdf_base64['data']
            except Exception as e:
                retry_count += 1
                logging.warning(f"PDF generation attempt {retry_count} failed: {str(e)}")
                if retry_count == max_retries:
                    raise RuntimeError(f"Failed to generate PDF after {max_retries} attempts")
                time.sleep(2)  # Wait before retrying
                
    except Exception as e:
        logging.error(f"Error in HTML_to_PDF: {str(e)}")
        raise RuntimeError(f"Failed to generate PDF: {str(e)}")
    finally:
        # Ensure driver is always quit
        if driver:
            try:
                driver.quit()
            except:
                pass
        # On Windows, we need to wait a bit for file handles to be released
        if os.name == 'nt':
            time.sleep(1)

def get_chrome_browser_options():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")  # Avvia il browser a schermo intero
    options.add_argument("--no-sandbox")  # Disabilita la sandboxing per migliorare le prestazioni
    options.add_argument("--disable-dev-shm-usage")  # Utilizza una directory temporanea per la memoria condivisa
    options.add_argument("--ignore-certificate-errors")  # Ignora gli errori dei certificati SSL
    options.add_argument("--disable-extensions")  # Disabilita le estensioni del browser
    options.add_argument("--disable-gpu")  # Disabilita l'accelerazione GPU
    options.add_argument("window-size=1200x800")  # Imposta la dimensione della finestra del browser
    options.add_argument("--disable-background-timer-throttling")  # Disabilita il throttling dei timer in background
    options.add_argument("--disable-backgrounding-occluded-windows")  # Disabilita la sospensione delle finestre occluse
    options.add_argument("--disable-translate")  # Disabilita il traduttore automatico
    options.add_argument("--disable-popup-blocking")  # Disabilita il blocco dei popup
    #options.add_argument("--disable-features=VizDisplayCompositor")  # Disabilita il compositore di visualizzazione
    options.add_argument("--no-first-run")  # Disabilita la configurazione iniziale del browser
    options.add_argument("--no-default-browser-check")  # Disabilita il controllo del browser predefinito
    options.add_argument("--single-process")  # Esegui Chrome in un solo processo
    options.add_argument("--disable-logging")  # Disabilita il logging
    options.add_argument("--disable-autofill")  # Disabilita l'autocompletamento dei moduli
    #options.add_argument("--disable-software-rasterizer")  # Disabilita la rasterizzazione software
    options.add_argument("--disable-plugins")  # Disabilita i plugin del browser
    options.add_argument("--disable-animations")  # Disabilita le animazioni
    options.add_argument("--disable-cache")  # Disabilita la cache
    #options.add_argument('--proxy-server=localhost:8081')
    #options.add_experimental_option("useAutomationExtension", False)  # Disabilita l'estensione di automazione di Chrome
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])  # Esclude switch della modalità automatica e logging

    options.add_argument("--single-process")  # Esegui Chrome in un solo processo
    return options

def printred(text):
    RED = "\033[91m"
    RESET = "\033[0m"
    print(f"{RED}{text}{RESET}")

def printyellow(text):
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    print(f"{YELLOW}{text}{RESET}")
