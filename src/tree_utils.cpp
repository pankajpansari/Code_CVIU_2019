#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <assert.h>
#include <Eigen/Dense>
#include "tree_utils.h"

//code for reading the tree from the short format, extracting information from it and printing the long format tree file

/////////////////////////////////
/// for reading graph ////
////////////////////////////////

void printVec(std::vector<int> a){
   for(int i = 0; i < a.size(); i++){
        std::cout << a[i] << "  "; 
   } 
    std::cout << std::endl;
}

node getRoot(std::vector<node> G){
    //given a tree, returns the node 
    for(int i = 0; i < G.size(); i++){
        if(G[i].parent == NULL)
            return G[i];
    }
}

void printTree(std::vector<node> G){

    //print root info
    node root = getRoot(G);
    std::cout << "Root id = " << root.id << std::endl;

    //for each node, print parent, children and weight from node to its parent

    for(int i = 0; i < G.size(); i++){
        std::cout << "Node id: " << G[i].id << std::endl; 
        if(G[i].id != root.id){
            std::cout << "Parent id: " << (*G[i].parent).id << std::endl; 
            std::cout << "Weight from node to parent : " << G[i].weight << std::endl; 
        }
        std::vector<node*> children = G[i].children;
        std::cout << "Children id: "; 
        for(int j = 0; j < children.size(); j++){
            std::cout << (*children[j]).id << "  "; 
        } 
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }
}

//void readTree(std::vector<std::vector<int>> &subleaves, std::vector<float> &node_weights, const std::string filename){
std::vector<node> readTree(const std::string filename){
    using namespace std;

    ifstream treefile(filename);
    string s;

    int nV;
    getline(treefile, s);
    istringstream ss(s);
    ss >> nV;

    vector<node> G(nV);
    for(int i = 0; i < G.size(); i++)
        G[i].id = i;

    int parent_id, child_id;
    float weight;

    while(getline(treefile, s)){
        istringstream ss(s);
        ss >> parent_id;
        ss >> weight;
        while(ss >> child_id){
            G[parent_id].children.push_back(&G[child_id]);
            G[child_id].weight = weight;
            G[child_id].parent = &G[parent_id];
        }
    }

    treefile.close();
    return G;
}


///////////////////////////////////////////
/// for extracting subtree information ////
//////////////////////////////////////////

//void getLeafNodes(node parent, std::vector<int> &leaf_vec)
void printLeafNodes(node parent)
{
    //returns the list of leaves of the subtree with parent as the root
    using namespace std;
    if(parent.children.size() == 0){
//        leaves.push_back(parent)
        std::cout << " " << parent.id;
        return;
    }
    else{
        for(int i = 0; i < parent.children.size(); i++){
            printLeafNodes(*parent.children[i]);
        }
    }
    return;
}

std::vector<node> getLeafNodes(const std::vector<node> &G)
{
    //returns the list of all the leaves of the tree
    std::vector<node> leaves;
    using namespace std;
    for(int i = 0; i < G.size(); i++){
        if(G[i].children.size() == 0){
            leaves.push_back(G[i]);
        }

    } 
    return leaves;
}

void getSubtreeLeafNodes(node parent, std::vector<node>* leaves, const std::vector<node> &G)
{
    //returns the list of leaves of the subtree with parent as the root

//  static std::vector<node> leaves;
    using namespace std;
    if(parent.children.size() == 0){
        (*leaves).push_back(parent);
    }
    else{
        for(int i = 0; i < parent.children.size(); i++){
            getSubtreeLeafNodes(*parent.children[i], leaves, G);
        }
    }
}

void checkLeafNodes(node parent, int M)
{
    //returns the list of leaves of the subtree with parent as the root
    using namespace std;
    if(parent.children.size() == 0){
        assert(parent.id >= 0 && parent.id < M && "Leaves in tree file should be numbered 0 - (#labels - 1)");
//        leaves.push_back(parent)
        return;
    }
    else{
        for(int i = 0; i < parent.children.size(); i++){
            checkLeafNodes(*parent.children[i], M);
        }
    }
    return;
}

int getLeafCount(node parent){
    using namespace std;
    static int count = 0;
    if(parent.children.size() == 0)
        count = count + 1;
    else{
        for(int i = 0; i < parent.children.size(); i++)
            getLeafCount(*parent.children[i]);
    } 
    return count;
}

//void getSubtrees(node parent, std::vector<std::vector<int>> &subleaves, std::vector<float> &node_weights, node root){
//    //get information about all subtrees in the tree
//    //for each subtree, gets the list of leaves and the upward root edge length
//    using namespace std;
//    vector<node*> children = parent.children;
//
//    for(int i = 0; i < children.size(); i++){
//        vector<int> leaf_vec;
//        getLeafNodes(*children[i], leaf_vec);
//        subleaves.push_back(leaf_vec);
//        get_node_weight(*children[i], node_weights, root);
//        getSubtrees(*children[i], subleaves, node_weights, root);
//    }
//    return;
//}

std::vector<node> getPath(node t){
    //given the tree G, and the leaf node t, get the list of nodes on the path from root to t
    node parent_node = t;
    std::vector<node> path;
    while(parent_node.parent != NULL){
        path.push_back(parent_node);
        parent_node = *parent_node.parent; 
    }
    return path;
}


void printLongTreeFile(const std::vector<node> &G){

    node root = getRoot(G);
    //print #meta-labels #labels
    std::cout << G.size() - 1 << " " << getLeafCount(root) << std::endl;
    
    //print path nodes
    for(int i = 0; i < G.size(); i++){
        if(G[i].id != root.id){
            std::cout << G[i].id;
            getPath(G[i]);
        } 
    }

    //print leaf nodes
    for(int i = 0; i < G.size(); i++){
        if(G[i].id != root.id){
            std::cout << G[i].id;
            printLeafNodes(G[i]);
            std::cout << std::endl;
        } 
    }

    //print weights
    for(int i = 0; i < G.size(); i++){
        if(G[i].id != root.id){
            float weight = G[i].weight;
            std::cout << G[i].id << " " << weight << std::endl;
        } 
    }
}

int getNumMetaLabels(const std::vector<node> &G){
    return G.size() - 1; //-1 to exclude the root
}

int getNumLabels(const std::vector<node> &G, int M){
    node root = getRoot(G);
    if( M != 0)
        checkLeafNodes(root, M);
    return getLeafCount(root);
}

Eigen::VectorXf getWeight(const std::vector<node> &G){
    int M = getNumMetaLabels(G);
    node root = getRoot(G);

    Eigen::VectorXf weight = Eigen::VectorXf::Zero(M);
    int label = 0;
    for(int i = 0; i < G.size(); i++){
        if(G[i].id != root.id){
            label = G[i].id;
            weight(label) = G[i].weight;
        } 
    }
    return weight;
}

float sumPath(node t, node s, const std::vector<node> &G){
    //adds up the edge weights on the upward path from t to s
    //assuming that s lies on path from root to t
    float edgeSum = 0;
    while(t.id != s.id){
        edgeSum += t.weight;
        t = *t.parent;
    }
    return edgeSum;
}

Eigen::MatrixXf getPairwiseTable(const std::vector<node> &G){
//print hierarchical Potts penalty among labels
//used for matlab trw
    std::vector<node> leaf_nodes = getLeafNodes(G);
    std::vector<node> sub_leaves;
    int L = getNumLabels(G);
    Eigen::MatrixXf compatibility_mat = Eigen::MatrixXf::Zero(L, L);
    node p;
    float ab_distance = 0;
    int break_flag = 0;
    for(int i = 0; i < leaf_nodes.size(); i++){
        node s = leaf_nodes[i];
//        std::cout << "s.id = " << s.id << std::endl;
        for(int j = 0; j < leaf_nodes.size(); j++){
            ab_distance = 0;
            node t = leaf_nodes[j];
            if(t.id == s.id){
                ab_distance = 0;
            }
 //           std::cout << "t.id = " << t.id << std::endl;
            else{
            p = s;
            break_flag = 0;
            sub_leaves.clear();
            while(true){
               ab_distance += p.weight;
 //           std::cout << "p.weight = " << p.weight << std::endl;
               p = *p.parent; 
               getSubtreeLeafNodes(p, &sub_leaves, G);
               for(int k = 0; k < sub_leaves.size(); k++){
                    if(sub_leaves[k].id == t.id)
                        break_flag = 1;  
                }
                if(break_flag == 1)
                    break;
            }
  //  std::cout << "break" << std::endl;
            p = t;
            break_flag = 0;
            sub_leaves.clear();
            while(true){
               ab_distance += p.weight;
   //         std::cout << "p.weight = " << p.weight << std::endl;
               p = *p.parent; 
               getSubtreeLeafNodes(p, &sub_leaves, G);
               for(int k = 0; k < sub_leaves.size(); k++){
                   if(sub_leaves[k].id == s.id)
                       break_flag = 1;  
               }
               if(break_flag == 1)
                   break;
            }
            }
            compatibility_mat(s.id, t.id) = ab_distance;
        }  
    }
    return compatibility_mat;
}

//int main(int argc, char *argv[]){
//    std::vector<node> G = readTree(argv[1]);
//    Eigen::MatrixXf m = getPairwiseTable(G);
//    std::cout << m << std::endl;
//    return 0;
//}
