import time
from testrunautolevelszip import upscale_zip_file

# Record the start time
start_time = time.time()

upscale_zip_file(r"C:\Users\jsoos\Documents\Calibre Library\Unknown\dl3pahxr (1096)\dl3pahxr - Unknown\OPS\c127", r"\\WEJJ-II\traiNNer-redux\experiments\4x_MangaJaNai_V1RC29_OmniSR\models\net_g_40000.pth")


# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Elapsed time: {elapsed_time:.2f} seconds")



