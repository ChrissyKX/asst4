#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
//#define VERBOSE 1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    int BUFSIZE = 64;
#pragma omp parallel
{
    double buf_frontier[BUFSIZE];
    int count = 0;
    double wtime = 0;
       
    #pragma omp for schedule(auto)
    for (int i=0; i<frontier->count; i++) {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        if (start_edge == end_edge) continue;
        
        // attempt to add all neighbors to the new frontier

        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER) {
                distances[outgoing] = distances[node] + 1;
                buf_frontier[count++] = outgoing;
                
                if (count == BUFSIZE) {
                    // flush the buffer

                    int old_index = new_frontier->count;
                    while (!__sync_bool_compare_and_swap(&(new_frontier->count), old_index, old_index + count)) {
                        old_index = new_frontier->count;
                    }

                    //double start = omp_get_wtime();
                    for (int j = old_index;j < old_index + count; j++) {
                        new_frontier->vertices[j] = buf_frontier[j - old_index];
                    }
                    //wtime += omp_get_wtime() - start;              
                    count = 0;
                }
            }
        }
         
    }

    if (count > 0) {
        int old_index = new_frontier->count;
        while (!__sync_bool_compare_and_swap(&(new_frontier->count), old_index, old_index + count)) {
            old_index = new_frontier->count;
        }
     
        for (int j = old_index;j < old_index + count; j++) {
            new_frontier->vertices[j] = buf_frontier[j - old_index];
        }
    }

    //wtime = omp_get_wtime() - wtime;
    //printf( "Time taken by thread %d is %f\n", omp_get_thread_num(), wtime);
}
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

int inline min(int a, int b) {
    return a < b ? a : b;
}

int bottom_up_step(Graph g, int* distances,
                    int frontier_version,
                   int unvisited_count, int* unvisited, int* new_unvisited)
{
    int new_unvisited_count = 0;
    int BUFSIZE = 64;
 
#pragma omp parallel
{
    int buf_unvisited[BUFSIZE];
    int count = 0;
    
    #pragma omp for schedule(auto)
    for (int i = 0; i < unvisited_count; i++) {
        int node = unvisited[i];
        
        int start_edge = g->incoming_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[node + 1];

        
        for (int neighbour = start_edge; neighbour < end_edge; neighbour++) {
            int incoming = g->incoming_edges[neighbour];
            if (distances[incoming] == frontier_version) {
                distances[node] = distances[incoming] + 1;
                                
                break;
            }
        }
        
        if (distances[node] == NOT_VISITED_MARKER) {
            buf_unvisited[count++] = node;

            if (count == BUFSIZE) {
                // flush the buffer
                int old_count = new_unvisited_count;
                while (!__sync_bool_compare_and_swap(&new_unvisited_count, old_count, old_count + count)) {
                    old_count = new_unvisited_count;    
                }
                
                for (int j = 0; j < count; j++) {
                     new_unvisited[old_count + j] = buf_unvisited[j];
                }
                count = 0;
            }          
        }
    }

     int old_count = new_unvisited_count;
     while (!__sync_bool_compare_and_swap(&new_unvisited_count, old_count, old_count + count)) {
         old_count = new_unvisited_count;    
     }
                
     for (int j = 0; j < count; j++) {
         new_unvisited[old_count + j] = buf_unvisited[j];
     }
}
    
    return new_unvisited_count;
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    int* frontier = (int*)malloc(sizeof(int) * graph->num_nodes);
    //int* new_frontier = (int*)malloc(sizeof(int) * graph->num_nodes); 
    int* unvisited = (int*)malloc(sizeof(int) * graph->num_nodes);  // dense unvisited set
    int* new_unvisited =  (int*)malloc(sizeof(int) * graph->num_nodes);

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
        if (i < ROOT_NODE_ID) {
            unvisited[i] = i;
        } else if (i > ROOT_NODE_ID) {
            unvisited[i - 1] = i;
        }
    }
    sol->distances[ROOT_NODE_ID] = 0;
    int unvisited_count = graph->num_nodes - 1;
        
    int frontier_count = 1;
    int frontier_version = 0;
    //frontier[ROOT_NODE_ID] = frontier_version; 
        
    while (frontier_count != 0) {
        int new_unvisited_count = bottom_up_step(graph, sol->distances,
                       frontier_version,
                       unvisited_count, unvisited, new_unvisited);

        frontier_version++;
        frontier_count = new_unvisited_count - unvisited_count;
        unvisited_count = new_unvisited_count;

        //swap pointers
        int* tmp = unvisited;
        unvisited = new_unvisited;
        new_unvisited = tmp;

        // tmp = frontier;
        // frontier = new_frontier;
        // new_frontier = tmp;

       
    }
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
