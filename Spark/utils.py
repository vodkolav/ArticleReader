
import zipfile, os, fnmatch

def zip_project(project_dir, zip_filename, exclude_patterns=None):
    exclude_patterns = exclude_patterns or []
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(project_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), project_dir)
                if file.endswith(".py") and not file.startswith(".") and not any(fnmatch.fnmatch(rel_path, p) for p in exclude_patterns):
                    zipf.write(os.path.join(root, file), rel_path)

    print(f"Created {zip_filename}")

if __name__ == "__main__": 
    zip_project(
        project_dir="ArticleReader",
        zip_filename="ArticleReader.zip",
        exclude_patterns=["trash/*"])