import streamlit.web.cli as stcli
import sys

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "C:\\Users\\nadez\\Desktop\\1_vyuka\\4 semestr\\ppy1\\pyMovie\\streamlit_app.py"]
    sys.exit(stcli.main())