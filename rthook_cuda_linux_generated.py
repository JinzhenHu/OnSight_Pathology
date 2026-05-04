
import os
import sys

if getattr(sys, 'frozen', False):

    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    

    lib_paths = [
        base_dir,
        os.path.join(base_dir, 'torch', 'lib'),
    ]
    
    nvidia_dir = os.path.join(base_dir, 'nvidia')
    if os.path.exists(nvidia_dir):
        for d in os.listdir(nvidia_dir):
            sub_lib = os.path.join(nvidia_dir, d, 'lib')
            if os.path.exists(sub_lib):
                lib_paths.append(sub_lib)
                

    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_paths) + (':' + current_ld if current_ld else '')
