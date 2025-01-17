#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{


  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs
  
   /*
     CS149 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
 
  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  
  double * score_old = (double *)std::malloc(sizeof(double) * numNodes);   // old scores
  double * score_new = (double *)std::malloc(sizeof(double) * numNodes);   // new scores
  Vertex * loners =    (Vertex *)std::malloc(sizeof(double) * numNodes);
  
  int num_loners = 0;
  #pragma omp parallel for 
  for (int v = 0; v < numNodes; v++) {
    score_old[v] = equal_prob;
    if (outgoing_size(g, v) == 0) {
      {
        #pragma omp critical 
        loners[num_loners++] = v;
      }
    }
  }

  
  /*
     CS149 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;
     
  */
  bool converged = false;

  // std::cout << "Starting the convergence loop" << std::endl;
  while (!converged) {

    double global_diff = 0.0;
    double sum_loners = 0.0;
    #pragma omp parallel for reduction(+:sum_loners)
    for (Vertex* v = loners; v < loners + num_loners; v++) {
        sum_loners += damping * score_old[*v] / numNodes; 
    }
    
    // compute score_new[vi] for all nodes vi:
    // score_new[vi] = sum over all nodes vj reachable from incoming edges
    //  { score_old[vj] / number of edges leaving vj  }
    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<numNodes; i++) {
      score_new[i] = 0;
      if (incoming_size(g, i) == 0) continue;
      // Vertex is typedef'ed to an int. Vertex* points into g.outgoing_edges[]
      const Vertex* start = incoming_begin(g, i);
      const Vertex* end = incoming_end(g, i);
      
      for (const Vertex* v=start; v!=end; v++) 
        score_new[i] += score_old[*v]/outgoing_size(g, *v);
      
    }

    
    // std::cout << "C1" << std::endl;
    // score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;
    #pragma omp parallel for 
    for (int i=0; i<numNodes; i++) {
      // Vertex is typedef'ed to an int. Vertex* points into g.outgoing_edges[]
   
      score_new[i] = (damping * score_new[i]) + (1.0 - damping) /  numNodes + sum_loners;
      /* }
    
    // std::cout << "C2" << std::endl;
    // score_new[vi] += sum over all nodes v in graph with no outgoing edges
    //                      { damping * score_old[v] / numNodes }
    #pragma omp parallel for 
    for (int i=0; i<numNodes; i++) {
      // Vertex is typedef'ed to an int. Vertex* points into g.outgoing_edges[]
      // if (i % 1000 == 0)
        // std::cout << "Doing outter " << i << std::endl;
        */
      // score_new[i] += sum_loners; 
      
    }
    // std::cout << "C3" << std::endl;
    // compute how much per-node scores have changed
    // quit once algorithm has converged
    

    // global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
    
    //  }
    #pragma omp parallel for reduction(+:global_diff)
    for (int i=0; i<numNodes; i++) {
      // Vertex is typedef'ed to an int. Vertex* points into g.outgoing_edges[]
      global_diff += std::abs(score_new[i] - score_old[i]);
    }
    // std::cout << global_diff << " " <<  convergence << std::endl;
    // converged = (global_diff < convergence)
    converged = global_diff < convergence;

    
    // This update should go last
    #pragma omp parallel for 
    for (int i=0; i<numNodes; i++) {
      // Vertex is typedef'ed to an int. Vertex* points into g.outgoing_edges[]
      score_old[i] = score_new[i];
    }
    
  }
  #pragma omp parallel for 
  for (int i=0; i<numNodes; i++) {
    // Vertex is typedef'ed to an int. Vertex* points into g.outgoing_edges[]
    solution[i] = score_new[i];
  }
  std::free(score_old);
  std::free(score_new);
  std::free(loners);
}
