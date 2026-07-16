from core.utils.logger import logger
import os
import sys
import pickle
import networkx as nx

try:
    from pyvis.network import Network
except ImportError:
    logger.info("Thư viện pyvis chưa được cài đặt. Đang tiến hành cài đặt...")
    os.system(f"{sys.executable} -m pip install pyvis --break-system-packages")
    from pyvis.network import Network


def visualize_graph(pkl_path, output_html):
    if not os.path.exists(pkl_path):
        logger.info(f"Lỗi: Không tìm thấy file {pkl_path}")
        return

    logger.info(f"Đang tải đồ thị từ {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    graph = data["graph"]
    nodes_data = data["nodes_data"]

    logger.info(f"Đồ thị có {graph.number_of_nodes()} đỉnh và {graph.number_of_edges()} cạnh.")

    # Khởi tạo mạng lưới pyvis
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        select_menu=True,
        filter_menu=True,
    )

    # Tùy chỉnh vật lý để đồ thị trải đều
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09,
        overlap=0,
    )

    logger.info("Đang thêm các đỉnh vào mạng lưới...")
    # Màu sắc dựa trên loại Node
    color_map = {
        "Cases": "#3b82f6",  # Xanh dương
        "Laws": "#10b981",  # Xanh lá
        "Crimes": "#ef4444",  # Đỏ
        "Cluster": "#f59e0b",  # Cam
    }

    for node_id in graph.nodes():
        node_type = nodes_data.get(node_id, {}).get("type", "Unknown")
        color = color_map.get(node_type, "#ffffff")

        # Tiêu đề khi hover
        title = f"Type: {node_type}\nID: {node_id}"

        # Nhãn hiển thị
        label = str(node_id)
        if len(label) > 15:
            label = label[:12] + "..."

        net.add_node(node_id, label=label, title=title, color=color, group=node_type)

    logger.info("Đang thêm các cạnh vào mạng lưới...")
    # Thêm cạnh
    for source, target, data in graph.edges(data=True):
        relation = data.get("relation_type", "")
        net.add_edge(source, target, title=relation)

    logger.info(f"Đang lưu file HTML tại: {output_html}")
    # Đổi thư mục hiện tại để pyvis tạo thư mục lib đúng vị trí
    original_cwd = os.getcwd()
    os.chdir(os.path.dirname(output_html))
    # Lưu file
    net.save_graph(os.path.basename(output_html))
    os.chdir(original_cwd)
    logger.info(f"Thành công! Hãy mở file {output_html} trên trình duyệt để xem.")


if __name__ == "__main__":
    pkl_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "processed",
        "graph.pkl",
    )
    html_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web", "graph_view.html"
    )
    visualize_graph(pkl_file, html_file)
