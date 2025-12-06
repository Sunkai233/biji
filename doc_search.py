#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档搜索工具 - 用于搜索 biji 仓库中的 Markdown 文档
"""

import os
import re
from pathlib import Path
from typing import List, Tuple


class DocSearch:
    """文档搜索类"""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.md_files = list(self.root_dir.glob("*.md"))

    def search_keyword(self, keyword: str, case_sensitive: bool = False) -> List[Tuple[str, int, str]]:
        """
        搜索关键词

        Args:
            keyword: 要搜索的关键词
            case_sensitive: 是否区分大小写

        Returns:
            [(文件名, 行号, 匹配行内容), ...]
        """
        results = []
        flags = 0 if case_sensitive else re.IGNORECASE

        for md_file in self.md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if re.search(keyword, line, flags):
                            results.append((md_file.name, line_num, line.strip()))
            except Exception as e:
                print(f"读取文件 {md_file} 时出错: {e}")

        return results

    def list_all_docs(self) -> List[str]:
        """列出所有 Markdown 文档"""
        return [f.name for f in self.md_files]

    def get_doc_stats(self) -> dict:
        """获取文档统计信息"""
        total_files = len(self.md_files)
        total_lines = 0
        total_size = 0

        for md_file in self.md_files:
            try:
                total_size += md_file.stat().st_size
                with open(md_file, 'r', encoding='utf-8') as f:
                    total_lines += sum(1 for _ in f)
            except Exception:
                pass

        return {
            "总文件数": total_files,
            "总行数": total_lines,
            "总大小(KB)": round(total_size / 1024, 2)
        }


def main():
    """主函数 - 命令行交互"""
    import sys

    searcher = DocSearch()

    if len(sys.argv) < 2:
        print("=" * 60)
        print("📚 文档搜索工具")
        print("=" * 60)

        # 显示统计信息
        stats = searcher.get_doc_stats()
        print(f"\n统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # 列出所有文档
        print(f"\n所有文档 ({len(searcher.md_files)} 个):")
        for i, doc in enumerate(searcher.list_all_docs(), 1):
            print(f"  {i}. {doc}")

        print("\n使用方法:")
        print("  python doc_search.py <关键词>")
        print("  例如: python doc_search.py Transformer")
        return

    # 搜索关键词
    keyword = sys.argv[1]
    results = searcher.search_keyword(keyword)

    print("=" * 60)
    print(f"🔍 搜索关键词: '{keyword}'")
    print("=" * 60)
    print(f"\n找到 {len(results)} 个匹配项:\n")

    current_file = None
    for filename, line_num, line_content in results:
        if filename != current_file:
            print(f"\n📄 {filename}")
            current_file = filename
        print(f"  第 {line_num} 行: {line_content[:100]}...")


if __name__ == "__main__":
    main()
