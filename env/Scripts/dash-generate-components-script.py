#!"c:\users\debabrata\documents\github\cuny-assignments\data 608 01[46846]\module4\heroku\dash_app_example\env\scripts\python.exe"
# EASY-INSTALL-ENTRY-SCRIPT: 'dash==1.9.1','console_scripts','dash-generate-components'
__requires__ = 'dash==1.9.1'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('dash==1.9.1', 'console_scripts', 'dash-generate-components')()
    )
