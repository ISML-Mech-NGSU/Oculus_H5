Oculusが出力するログファイル.oculusを、MATLABで読み込み可能なH5形式に変換するツールです。<br>
MATLABでの読み込みサンプルもあります<br>
<br>
Python スクリプトには、以下のモジュール読み込みを行っています。<br>
import sqlite3
import struct
import zlib
import sys
from pathlib import Path
import h5py
import numpy as np