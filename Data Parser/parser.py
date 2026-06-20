import os

def clean_file(filename):
    non_text_message = False
    data = []
    with open(filename, encoding="utf-8") as file:
        for line in file.readlines()[5:-4]:
            # Format author tag
            if line.startswith("["):
                line = ("["+line[line.find("]") + 2:-1]+"]: ")
                line = line.replace("abyssalsand", "me")

            # Cuts out non-message data
            if line.startswith("{"):
                if data[-1].startswith("["):
                    data.pop()
                non_text_message = True
            if non_text_message:
                if line != "\n":
                    continue
                else:
                    non_text_message = False

            # Cuts out newlines
            if line == "\n":
                continue
            
            data.append(line)
    username = filename[filename.find("-") + 2:filename.find("[") - 1]
    clean_filepath = filename[:filename.find("Data\\")] + "CleanedData\\" + username + ".txt"
    with open(clean_filepath, "w", encoding="utf-8") as file:
        for line in data:
            file.write(line)

if __name__ == "__main__":
    src_dir = "D:\\Visual Studio\\Rust\\LLM\\Data Parser\\Data"
    for file in os.listdir(src_dir):
        clean_file(src_dir + "\\" + file)
