package cn.edu.nuaa.cluster;

public class Configuration {
	
	public static final String ROOT_PATH = "output/"; // The root path of all output data.

	// the input path of fix patterns mining.
	private static final String MINING_INPUT = ROOT_PATH + "MiningInput/";
	public static final String CLUSTER_INPUT = MINING_INPUT + "ClusteringInput/input.arff";

	// the output path of fix patterns mining.
	private static final String MINING_OUTPUT = ROOT_PATH + "MiningOutput/";
	public static final String CLUSTER_OUTPUT = MINING_OUTPUT + "ClusteringOutput/clusterResults.list";

}
