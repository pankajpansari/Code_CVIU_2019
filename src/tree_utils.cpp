#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <vector>
#include <assert.h>

//code for reading the tree from the short format, extracting information from it and printing the long format tree file

struct node{
    int id;
    node* parent;
    std::vector<node*> children;
    float weight;
};

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
    return G;
}


///////////////////////////////////////////
/// for extracting subtree information ////
//////////////////////////////////////////

//void getLeafNodes(node parent, std::vector<int> &leaf_vec)
void getLeafNodes(node parent)
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
            getLeafNodes(*parent.children[i]);
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

void getPath(node t){
    //given the tree G, and the leaf node t, get the list of nodes on the path from root to t
    node parent_node = t;
//    std::cout << "Path from root to node id " << t.id << ":     ";
    while(parent_node.parent != NULL){
        std::cout << " " << parent_node.id;
//        path_nodes.push_back(parent_node);
        parent_node = *parent_node.parent; 
    }
    std::cout << std::endl;
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
            getLeafNodes(G[i]);
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

int main(int argc, char *argv[]){
    std::vector<node> G = readTree(argv[1]);
    printLongTreeFile(G);
    return 0;
}
