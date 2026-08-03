'''
Chroma Colletction 清理脚本

    目的： 安全列出chorma_db中的所有collection，并且只删除指定名字的collection，避免误删其他collection
    原理：chroma持久化目录下，每个collection对应一个UUID的子目录，删除collection就是删除对应的子目录，
        存储在chroma的元数据库里，不是文件夹名本身。需要通过chroma的API来获取collection名对应的UUID，才能安全删除。
'''
import argparse
import chromadb
from config.settings import settings

def list_collections():
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collections = client.list_collections() # 返回一个列表，每个元素是一个dict，包含collection的name和id

    if not collections:
        print("当前chroma_db中没有任何collection")
        return
    print(f"当前的chormadb中有 {len(collections)} 个collection:")

    for col in collections:
        print(f"collection name: {col.name}, collection id: {col.id}")

def delete_collection(collection_name:str):
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    existing_collections =  [col.name for col in client.list_collections()]

    if collection_name not in existing_collections:
        print(f"collection {collection_name} 不存在，无法删除")
        return

    # 删除指定collection
    client.delete_collection(name=collection_name)
    print(f"collection {collection_name} 已删除")

    print("删除后，当前的chormadb中有以下collection:")
    list_collections()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chroma Collection 清理脚本")
    parser.add_argument("--list",action="store_true",help="列出当前chroma_db中的所有collection")
    parser.add_argument("--delete",type=str,help="删除指定名字的collection")
    args = parser.parse_args()  # 解析命令行参数

    if args.list:
        list_collections()
    elif args.delete:
        delete_collection(args.delete)
    else:
        print("请使用 --list 或 --delete <collection_name> 参数")

