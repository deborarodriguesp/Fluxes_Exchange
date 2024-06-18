**QGIS preparation**

1. Necessary Files
   Hydrological DTM
   Hydrodynamic DTM
   
3. Steps:
   Create polygon and then outside lines of the Hydrodinamic dtm grid

   Cut the lines to the study area and remove any islands

   Use v.parallel from GRASS (QGIS) to set a buffer from the initial line and create a parallel line.

   For each line, initial and parallel, select the elements of the hydrological DTM through localization,
   and save only the selected elements

   Create centroids to this file with selected elements, and atribute geometry
   to add coordinates to the file, and save this last file.

The last file, with coodinates, will be read by the python script. 
   
