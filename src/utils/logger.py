import os
import sys
sys.stdout.reconfigure(encoding='utf-8')


class Logger:
    @staticmethod
    def section(title: str):
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
    
    @staticmethod
    def info(msg: str):
        print(f"\n{msg}")
    
    @staticmethod
    def progress(msg: str):
        print(msg)
