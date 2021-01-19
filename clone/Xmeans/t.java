package edu.lu.uni.serval.cluster;

//import edu.lu.uni.serval.utils.FileHelper;
//import edu.lu.uni.serval.config.Configuration;



//import weka.clusterers.XMeans;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.clusterers.XMeans;
import weka.core.ChebyshevDistance;
import weka.core.DistanceFunction;
import weka.core.Instances;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;