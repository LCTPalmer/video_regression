from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool, PanTool, WheelZoomTool
import numpy as np
import struct, os

#bokeh settings
output_file("load_video_vis.html")
figure_size = 600

#Prepare the data
video_dir = './' #directory of videos
mapping_file = './bokeh_mds_mapping.csv' #path to csv file

#initialise data lists
X, Y, ratings, video_names = [], [], [], []
#read in the csv
with open(mapping_file, 'rb') as f:
    import csv
    cr = csv.reader(f)
    for row in cr:
        video_names.append(row[0])
        ratings.append(row[1])
        X.append(row[2])
        Y.append(row[3])

# ##toy example--------------------------------------------------
# X = [1, 2, 3, 4, 5] #MDS component 1
# Y = [2, 5, 8, 2, 7] #MDS component 2
# ratings = [2,5,4,6,10] #trueskill ratings
# video_names = ["v1.ogv", "v1.ogv", "v1.ogv", "v1.ogv", "v1.ogv"]
# ##------------------------------------------------------------

#turn video names into full path for playing
video_paths = [os.path.join(video_dir, name) for name in video_names]
#other information to display on hover
time_stamps = [video_name[:19] for video_name in video_names]

#put into Bokeh format
source = ColumnDataSource(
        data=dict(
            x=X,
            y=Y,
            video_path=video_paths,
            time_stamp=time_stamps,
            rating=ratings
        )
    )

#HTML for displaying the video on hover
hover = HoverTool(
        tooltips="""
        <div>
            <span style="font-size: 17px;">load rating:</span>
            <span style="font-size: 15px;">@rating</span>
        </div>
        <div>
            <span style="font-size: 17px;">recorded at:</span>
            <span style="font-size: 15px;">@time_stamp</span>
        </div>
        <div>
            <video width="300" autoplay loop>
                <source src=@video_path type="video/mp4">
            </video>
        </div>
        """
    )

#set up the plot
p = figure(plot_width=figure_size, plot_height=figure_size, tools=[hover, PanTool(), WheelZoomTool()], 
           title="Perceptual load in driving videos")

#map to colors from ratings
min_rating = min(ratings); max_rating = max(ratings)
ratings_norm = [(rating - min_rating) / float(max_rating - min_rating) for rating in ratings]
colors = [rating*255 for rating in ratings_norm]
colors_rgb = [(color,color,color) for color in colors]
colors_hex = ['#' + struct.pack('BBB',*color_rgb).encode('hex') for color_rgb in colors_rgb]

#draw the scatter plot
p.circle('x', 'y', size=10, source=source, fill_color=colors_hex, line_color='black', alpha=0.7)

#output
show(p)
