
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <utility>
#include <iostream>

namespace py = pybind11;

class SumTree {
private:
    size_t capacity;
    size_t write;
    std::vector<double> tree;
    std::vector<py::object> data;

    void propagate(size_t idx, double change) {
        while (idx != 0) {
            size_t parent = (idx - 1) / 2;
            tree[parent] += change;
            idx = parent;
        }
    }

    size_t retrieve(size_t idx, double val) const {
        while (true) {
            size_t left = 2 * idx + 1;
            if (left >= tree.size()) return idx;
            if (val <= tree[left]) {
                idx = left;
            }
            else {
                idx = left + 1;
                val -= tree[left];
            }
        }
    }

public:
    SumTree() : capacity(0), write(0) {}    //デフォルトコンストラクタ

    SumTree(size_t capacity)
        : capacity(capacity), write(0) {
        tree.resize(2 * capacity - 1, 0.0);
        data.resize(capacity, py::none());
    }

    double total() const {
        return tree[0];
    }

    void add(double priority, const py::object& dat) {
        size_t tree_idx = write + capacity - 1;
        data[write] = dat;
        update(tree_idx, priority);
        write = (write + 1) % capacity;
    }

    void update(size_t tree_idx, double priority) {
        double change = priority - tree[tree_idx];
        tree[tree_idx] = priority;
        propagate(tree_idx, change);
    }

    std::tuple<size_t, double, py::object> get(double val) const {
        size_t idx = retrieve(0, val);
        size_t data_idx = idx - capacity + 1;
        return std::make_tuple(idx, tree[idx], data[data_idx]);
    }

    size_t get_write() const { return write; }
    void set_write(size_t w) { write = w; }

    const std::vector<double>& get_tree_vector() const { return tree; }
    
    void set_tree_vector(const std::vector<double>& t) { tree = t; }

    const std::vector<py::object>& get_data_vector() const { return data; }
    
    void set_data_vector(const std::vector<py::object>& d) { data = d; }
};

class ProportionalMemory {
private:
    size_t capacity;
    SumTree tree;
    double beta_initial;
    double beta_steps;
    double epsilon;
    double alpha;
    bool has_duplicate;
    double max_priority;
    size_t size;

public:
    ProportionalMemory(size_t capacity,
                       double alpha = 0.6,
                       double beta_initial = 0.4,
                       double beta_steps = 1000000,
                       bool has_duplicate = true,
                       double epsilon = 0.0001)
        : capacity(capacity),
          beta_initial(beta_initial), beta_steps(beta_steps),
          epsilon(epsilon), alpha(alpha), has_duplicate(has_duplicate),
          max_priority(1.0), size(0)
    {
        clear();
    }

    void clear() {
        tree = SumTree(capacity);
        max_priority = 1.0;
        size = 0;
    }

    size_t length() const {
        return size;
    }

    void add(const py::object& batch, std::optional<double> priority = std::nullopt, bool restore_skip = false) {
        double final_priority;

        if (!priority.has_value()) {
            final_priority = max_priority;
        } else if (!restore_skip) {
            final_priority = std::pow(std::abs(priority.value()) + epsilon, alpha);
        } else {
            final_priority = priority.value();
        }

        tree.add(final_priority, batch);

        size += 1;
        if (size > capacity) {
            size = capacity;
        }
    }

    py::tuple sample(size_t batch_size, size_t step) {
        std::vector<size_t> indices;
        std::vector<py::object> batches;
        std::vector<float> weights(batch_size, 0.0f);

        double total = tree.total();

        // βは最初は低く、学習終わりに1にする
        double beta = beta_initial + (1.0 - beta_initial) * static_cast<double>(step) / static_cast<double>(beta_steps);
        if (beta > 1.0) beta = 1.0;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, total);

        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = 0;
            double priority = 0.0;
            py::object batch;

            for (size_t tries = 0; tries < 9999; ++tries) {
                double r = dis(gen);
                auto [idx_, priority_, batch_] = tree.get(r);

                // 重複を許可しない場合はやり直す
                if (!has_duplicate) {
                    bool duplicate = false;
                    for (const auto& existing_idx : indices) {
                        if (existing_idx == idx_) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (duplicate) {
                        continue;  // やり直し
                    }
                }

                idx = idx_;
                priority = priority_;
                batch = batch_;
                break;  // OKなら抜ける
            }

            indices.push_back(idx);
            batches.push_back(batch);

            // 重要度サンプリングを計算 w = (N * pi)
            double prob = priority / total;
            weights[i] = static_cast<float>(std::pow(static_cast<double>(size) * prob, -beta));
        }

        // 最大値で正規化
        float max_weight = *std::max_element(weights.begin(), weights.end());
        if (max_weight > 0.0f) {
            for (auto& w : weights) {
                w /= max_weight;
            }
        }
        return py::make_tuple(batches, weights, indices);
    }

    void update(const std::vector<size_t>& indices, const std::vector<float>& priorities) {
        for (size_t i = 0; i < indices.size(); ++i) {
            double priority = std::pow(std::abs(static_cast<double>(priorities[i])) + epsilon, alpha);
            tree.update(indices[i], priority);
             if(max_priority < priority) {max_priority = priority;}
        }
    }

    py::list backup() const {
        py::list dat;
        dat.append(capacity);
        dat.append(max_priority);
        dat.append(size);
        dat.append(tree.get_write());
        dat.append(tree.get_tree_vector());
        dat.append(tree.get_data_vector());
        return dat;
    }

    void restore(const py::list& data) {
        if (capacity == data[0].cast<size_t>()) {
            max_priority = data[1].cast<double>();
            size = data[2].cast<size_t>();
            tree.set_write(data[3].cast<size_t>());
            tree.set_tree_vector(data[4].cast<std::vector<double>>());
            tree.set_data_vector(data[5].cast<std::vector<py::object>>());
        } else {
            clear();
            size_t new_capacity = data[0].cast<size_t>();
            size_t new_size = data[2].cast<size_t>();
            std::vector<double> tree_vec = data[4].cast<std::vector<double>>();
            std::vector<py::object> tree_data = data[5].cast<std::vector<py::object>>();
            for (size_t i = 0; i < new_size; ++i) {
                py::object d = tree_data[i];
                double priority = tree_vec[i + new_capacity - 1];
                add(d, priority, true);  // restore用: _restore_skip=true 相当
            }
        }
    }
};

PYBIND11_MODULE(proportional_memory_cpp, m) {
    py::class_<ProportionalMemory>(m, "ProportionalMemory")
        .def(py::init<size_t, double, double, size_t, bool, double>(),
             py::arg("capacity"),
             py::arg("alpha") = 0.6,
             py::arg("beta_initial") = 0.4,
             py::arg("beta_steps") = 1000000,
             py::arg("has_duplicate") = true,
             py::arg("epsilon") = 0.0001)
        .def("clear", &ProportionalMemory::clear)
        .def("length", &ProportionalMemory::length)
        .def("add",
            [](ProportionalMemory& self, const py::object& batch,
            std::optional<double> priority, bool restore_skip) {
            self.add(batch, priority, restore_skip);
            },
            py::arg("batch"),
            py::arg("priority") = py::none(),
            py::arg("restore_skip") = false)
        .def("sample", &ProportionalMemory::sample,
            py::arg("batch_size"),
            py::arg("step"))
        .def("update", &ProportionalMemory::update)
        .def("backup", &ProportionalMemory::backup)
        .def("restore", &ProportionalMemory::restore);
}