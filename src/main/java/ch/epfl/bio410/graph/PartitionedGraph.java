package ch.epfl.bio410.graph;


import ij.ImagePlus;
import ij.gui.Line;
import ij.gui.OvalRoi;
import ij.gui.Overlay;

import java.util.ArrayList;


/**
 * Class implementing a "PartitionedGraph" object.  A PartitionedGraph represents a collection of Spots objects, where
 * each Spots list corresponds to a single trajectory. This class is useful in particle tracking to manage, visualize,
 * and analyze multiple trajectories.
 */
public class PartitionedGraph extends ArrayList<Spots> {


    /**
     * This method retrieves the trajectory that contains the given Spot object.
     *
     * @param spot The Spot to search for.
     * @return The Spots list that contains the specified spot, or null if not found.
     */
    public Spots getPartitionOf(Spot spot) {
        for (Spots spots : this) {
            for (Spot test : spots) {
                if (spot.equals(test))
                    return spots;
            }
        }
        return null;
    }


    /**
     * This method creates a new trajectory containing a single Spot and adds it to the graph.
     *
     * @param spot The initial Spot of the new trajectory.
     * @return The created Spots list.
     */
    public Spots createPartition(Spot spot) {
        Spots spots = new Spots();
        spots.add(spot);
        add(spots);
        return spots;
    }


    /**
     * This method draws circular ROIs for each Spot object in all trajectories and adds them to an overlay. Each spot
     * is drawn as a small oval at its (x, y) position on its corresponding time frame.
     *
     * @param imp The ImagePlus object to which the spots are drawn.
     * @return The Overlay containing all spot ROIs.
     */
    public Overlay drawSpots(ImagePlus imp) {
        // get the existing overlay from the image, or create a new one
        Overlay overlay = imp.getOverlay();
        if (overlay == null) overlay = new Overlay();

        int radius = 5; // fixed radius for each spot circle
        for(Spots spots : this) { // loop through all trajectories
            for(Spot spot : spots) {
                // position the circle centered on the spot's coordinates
                double xp = spot.x + 0.5 - radius;
                double yp = spot.y + 0.5 - radius;
                OvalRoi roi = new OvalRoi(xp, yp, 2 * radius, 2 * radius);
                roi.setPosition(spot.t + 1); // assign ROI to display only on the corresponding frame
                // style the ROI with trajectory-specific color
                roi.setStrokeColor(spots.color);
                roi.setStrokeWidth(1);
                overlay.add(roi);
            }
        }
        // duplicate the input image for display with the overlay
        ImagePlus out = imp.duplicate();
        out.setTitle("Spots " + imp.getTitle() );
        out.show();
        out.getProcessor().resetMinAndMax();
        out.setOverlay(overlay);
        return overlay;
    }


    /**
     * This method draws line segments between consecutive spots in each partition to visualize trajectories. It also
     * draws circular ROIs at each spot location at the current frame. It can optionally use speed-based coloring for
     * each trajectory and ROI.
     *
     * @param imp The ImagePlus object to which the lines and ROIs are drawn.
     * @param withSpeed If true, color lines and spots based on per-frame speed color; otherwise use the base color.
     * @return The Overlay containing all line and spot ROIs.
     */
    public Overlay drawLines(ImagePlus imp, Boolean withSpeed) {
        // get the existing overlay from the image, or create a new one
        Overlay overlay = imp.getOverlay();
        if (overlay == null) overlay = new Overlay();

        int radius = 3;
        for (Spots spots : this) {
            if (spots.isEmpty()) break;

            for (int i = 1; i < spots.size(); i++) { // iterate through pairs of consecutive spots
                Spot spot = spots.get(i);
                Spot prev = spots.get(i - 1);
                // create a line from previous spot to current spot
                Line line = new Line(prev.x + 0.5, prev.y + 0.5, spot.x + 0.5, spot.y + 0.5);

                // set color depending on whether speed coloring is enabled
                if(withSpeed){
                    line.setStrokeColor(spots.speed_color[i]);
                }else{
                    line.setStrokeColor(spots.color);
                }

                line.setStrokeWidth(2);
                overlay.add(line);

                // draw ROI at current spot depending on whether speed coloring is enabled
                OvalRoi roi1 = new OvalRoi(spot.x + 0.5 - radius, spot.y + 0.5 - radius, 2 * radius, 2 * radius);
                roi1.setPosition(spot.t + 1);
                if(withSpeed){
                    roi1.setFillColor(spots.speed_color[i]);
                }else{
                    roi1.setFillColor(spots.color);
                }

                roi1.setStrokeWidth(1);
                overlay.add(roi1);

                // draw ROI at previous spot depending on whether speed coloring is enabled
                OvalRoi roi2 = new OvalRoi(prev.x + 0.5 - radius, prev.y + 0.5 - radius, 2 * radius, 2 * radius);
                roi2.setPosition(prev.t + 1);
                if(withSpeed){
                    roi2.setFillColor(spots.speed_color[i]);
                }else{
                    roi2.setFillColor(spots.color);
                }
                roi2.setStrokeWidth(1);
                overlay.add(roi2);
            }
        }
        // use the original image directly for ROI display to avoid memory issues with duplication
        ImagePlus out = imp;
        out.setTitle("Trajectories " + imp.getTitle() );
        out.show();
        out.getProcessor().resetMinAndMax();
        out.setOverlay(overlay);
        return overlay;
    }
}
