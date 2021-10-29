struct trie_node {
    bool is_end; map<char, trie_node*> nodes;
    trie_node() { is_end = false; }
};

class trie {
    bool remove_leaf(trie_node* node) {
        if (!node->is_end && node->nodes.empty()) {
            delete node;
            return true;
        }
        return false;
    }

    bool remove_rec(trie_node* node, string& key, int depth) {
        if (depth == key.size()) {
            node->is_end = false;
            return remove_leaf(node);
        }
        auto search = node->nodes.find(key[depth]);
        if (search == node->nodes.end() ||
            !remove_rec(search->second, key, depth + 1))
            return false;
        node->nodes.erase(search);
        return remove_leaf(node);
    }
    trie_node* root;

public:
    trie()
        : root(new trie_node()) {}
    void insert(string key) {
        trie_node* node = root;
        for (char c : key) {
            if (node->nodes.find(c) == node->nodes.end())
                node->nodes[c] = new trie_node();
            node = node->nodes[c];
        }
        node->is_end = true;
    }
    bool search(string key) {
        trie_node* node = root;
        for (char c : key) {
            if (node->nodes.find(c) == node->nodes.end())
                return false;
            node = node->nodes[c];
        }
        return node->is_end;
    }
    inline void remove(string key) { remove_rec(root, key, 0); }
};