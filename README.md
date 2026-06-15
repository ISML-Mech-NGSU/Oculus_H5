Oculusが出力するログファイル.oculusを、MATLABで読み込み可能なH5形式に変換するツールです。<br>
MATLABでの読み込みサンプルもあります<br>
<br>
Python スクリプトには、以下のモジュール読み込みを行っています。<br>
import sqlite3<br>
import struct<br>
import zlib<br>
import sys<br>
from pathlib import Path <br>
import h5py <br>
import numpy as np <br>