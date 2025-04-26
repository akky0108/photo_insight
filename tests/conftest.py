import sys
import os

# tests/ 配下から見て、プロジェクトルートと src をパスに追加
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_root  = os.path.join(proj_root, 'src')

# どちらも先頭に置く
sys.path.insert(0, src_root)
sys.path.insert(0, proj_root)
