import dropbox

def download_file_from_shared_link(access_token, shared_link, local_file_path):
    """
    Downloads a file from Dropbox using a shared link.
    
    Args:
    - access_token (str): Your Dropbox API access token.
    - shared_link (str): The shared link URL of the Dropbox file.
    - local_file_path (str): The local path where the file will be saved.
    """
    # Create a Dropbox client
    dbx = dropbox.Dropbox(access_token)

    try:
        # Download the file using the shared link
        metadata, response = dbx.sharing_get_shared_link_file(url=shared_link)

        # Write the file content to the specified local path
        with open(local_file_path, "wb") as file:
            file.write(response.content)

        print(f"File downloaded successfully to: {local_file_path}")

    except dropbox.exceptions.ApiError as error:
        print(f"Error downloading file: {error}")

if __name__ == "__main__":
    # Replace this with your Dropbox access token
    ACCESS_TOKEN = "
    # Dropbox shared link (make sure it includes the ?dl=0 parameter for downloading)
    SHARED_LINK = "https://www.dropbox.com/scl/fi/359jougtvzkohzokrbs64/hybrid_csv_Jabil_train.pkl?rlkey=r97azj0fyzt8k0wmpegi6mljk&dl=0"

    # Local path to save the file
    LOCAL_FILE_PATH = "/root/pyskl_thesis/hybrid_train_Ja.pkl"

    # Call the function
    download_file_from_shared_link(ACCESS_TOKEN, SHARED_LINK, LOCAL_FILE_PATH)

