import os.path    
import cgi, cgitb 
import joblib     
import sys, codecs
from pydoc import html 


# SCRIPT_MODE = True    
# cgitb.enable()      


def showHTML(text, msg):
    print("Content-Type: text/html; charset=utf-8")
    print(f"""
    
        <!DOCTYPE html>
        <html lang="en">
         <head>
          <meta charset="UTF-8">
          <title>---AI언어판별---</title>
         </head>
         <body>
          <form>
            <textarea name="text" rows="10" colos="40" >{text}</textarea>
            <p><input type="submit" value="언어감지">{msg}</p>
          </form>
         </body>
        </html>""")

# if SCRIPT_MODE:
#     sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())


text = "Hello"
msg =""

showHTML(text,msg)