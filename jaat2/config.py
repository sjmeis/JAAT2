import logging
import os

logger = logging.getLogger("JAATv2")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

GLOBAL_SETTINGS = {
    "show_progress": True,
    "single_threaded": False
}

MODEL_CACHE = {}

VERSION = "2.0.3beta"
COPYRIGHT = "© 2026 Stephen Meisenbacher. All rights reserved."
DESCRIPTION = "An NLP-powered toolkit for job ad text analysis."

CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Note: 3D Diagonal
JAAT_ART = r"""
                                                     ,----,                      
         ,---._                                    ,/   .`|                      
       .-- -.' \    ,---,         ,---,          ,`   .'  :             ,----,   
       |    |   :  '  .' \       '  .' \       ;    ;     /           .'   .' \  
       :    ;   | /  ;    '.    /  ;    '.   .'___,/    ,'          ,----,'    | 
       :        |:  :       \  :  :       \  |    :     |     .---. |    :  .  ; 
       |    :   ::  |   /\   \ :  |   /\   \ ;    |.';  ;   /.  ./| ;    |.'  /  
       :         |  :  ' ;.   :|  :  ' ;.   :`----'  |  | .-' . ' | `----'/  ;   
       |    ;   ||  |  ;/  \   \  |  ;/  \   \   '   :  ;/___/ \: |   /  ;  /    
   ___ l         '  :  | \  \ ,'  :  | \  \ ,'   |   |  '.   \  ' .  ;  /  /-,   
 /    /\    J   :|  |  '  '--' |  |  '  '--'     '   :  | \   \   ' /  /  /.`|   
/  ../  `..-    ,|  :  :       |  :  :           ;   |.'   \   \  ./__;      :   
\    \         ; |  | ,'       |  | ,'           '---'      \   \ |   :    .'    
 \    \      ,'  `--''         `--''                         '---";   | .'       
  "---....--'                                                     `---'          
                                                                                           
"""
SHOW_ART = os.isatty(1)
