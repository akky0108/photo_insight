import os
import yaml
from github import Github
from github.GithubException import GithubException
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()
token = os.getenv("GITHUB_TOKEN")
if not token:
    raise ValueError("環境変数 GITHUB_TOKEN が設定されていません")

# GitHub 認証とリポジトリ取得
g = Github(token)
repo_name = "akky0108/photo_insight"
repo = g.get_repo(repo_name)

# issues.yml の読み込み
script_dir = os.path.dirname(os.path.abspath(__file__))
issues_file = os.path.join(script_dir, "issues.yml")

with open(issues_file, "r") as f:
    data = yaml.safe_load(f)

# 既存ラベルを取得
existing_labels = {label.name: label for label in repo.get_labels()}

# イシュー作成
for issue in data["issues"]:
    title = issue["title"]
    body = issue.get("body", "")
    label_names = issue.get("labels", [])
    label_objects = []

    for label_name in label_names:
        if label_name not in existing_labels:
            print(f"ラベル '{label_name}' が存在しません。作成します。")
            try:
                new_label = repo.create_label(name=label_name, color="f29513")
                existing_labels[label_name] = new_label
            except GithubException as e:
                print(f"ラベル作成に失敗: {label_name} - エラー: {e}")
                continue
        label_objects.append(existing_labels[label_name])

    try:
        repo.create_issue(title=title, body=body, labels=label_objects)
        print(f"Issue 作成成功: {title}")
    except GithubException as e:
        print(f"Issue 作成失敗: {title} - エラー: {e}")
