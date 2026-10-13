import json
import streamlit as st
import streamlit.components.v1 as components

def get_3d_node_state(t_data):
    st_val = t_data.get("status", "ACTIVE")
    health_val = t_data.get("health", "GOOD")
    loss = t_data.get("packet_loss_pct", 0.0)
    snr = t_data.get("snr_db", 20.0)
    
    if st_val == "INACTIVE":
        return "#64748B", "INACTIVE"
    elif health_val in ["EXCELLENT", "GOOD"] and loss < 3.0 and snr >= 12.0:
        return "#10B981", "GOOD"
    elif health_val == "DEGRADED" or (3.0 <= loss < 8.0) or (7.0 <= snr < 12.0):
        return "#F59E0B", "MODERATE"
    else:
        return "#EF4444", "POOR"

def render_3d_digital_twin(node_telemetry, selected_dest_nodes=None, height=480):
    if selected_dest_nodes is None:
        selected_dest_nodes = ["N1", "N2", "N3"]
        
    formatted_data = {}
    for nid, t_data in node_telemetry.items():
        color_hex, state_label = get_3d_node_state(t_data)
        is_target = nid in selected_dest_nodes
        
        # Delivery status calculation applies ONLY to target nodes
        if is_target:
            deliv_st = t_data.get("delivery_status", "DELIVERED" if t_data.get("status") == "ACTIVE" else "FAILED")
        else:
            deliv_st = "N/A (not targeted)"

        formatted_data[nid] = {
            "status": t_data.get("status", "ACTIVE"),
            "health": t_data.get("health", "GOOD"),
            "snr_db": float(t_data.get("snr_db", 20.0)),
            "latency_ms": float(t_data.get("latency_ms", 10.0)),
            "packet_loss_pct": float(t_data.get("packet_loss_pct", 0.0)),
            "delivery_status": deliv_st,
            "isTarget": is_target,
            "colorHex": color_hex,
            "stateLabel": state_label
        }
        
    json_node_data = json.dumps(formatted_data)
    json_selected = json.dumps(selected_dest_nodes)
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                overflow: hidden;
                background-color: #0F172A;
                font-family: 'Inter', system-ui, sans-serif;
                user-select: none;
            }}
            #canvas-container {{
                width: 100vw;
                height: 100vh;
                position: relative;
            }}
            #tooltip {{
                position: absolute;
                display: none;
                background: rgba(15, 23, 42, 0.94);
                border: 1px solid #38BDF8;
                border-radius: 8px;
                padding: 10px 14px;
                color: #F8FAFC;
                font-size: 12px;
                pointer-events: none;
                box-shadow: 0 4px 14px rgba(0,0,0,0.5);
                z-index: 100;
                line-height: 1.4;
            }}
            .tt-title {{
                font-weight: 700;
                font-size: 13px;
                color: #38BDF8;
                margin-bottom: 4px;
            }}
            .tt-badge {{
                display: inline-block;
                padding: 2px 6px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 11px;
            }}
            #controls-hint {{
                position: absolute;
                bottom: 10px;
                left: 12px;
                color: #94A3B8;
                font-size: 11px;
                background: rgba(15, 23, 42, 0.75);
                padding: 4px 10px;
                border-radius: 6px;
                border: 1px solid #334155;
                pointer-events: none;
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="canvas-container">
            <div id="tooltip"></div>
            <div id="controls-hint">🖱️ Left-click drag: Rotate 3D | Scroll: Zoom | Right-click: Pan | Hover node: Inspect</div>
        </div>
        <script>
            const nodeData = {json_node_data};
            const selectedNodes = {json_selected};

            const container = document.getElementById('canvas-container');
            const tooltip = document.getElementById('tooltip');

            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0F172A);

            const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 7, 16);

            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
            scene.add(ambientLight);

            const dirLight = new THREE.DirectionalLight(0x38BDF8, 0.9);
            dirLight.position.set(10, 20, 15);
            scene.add(dirLight);

            const pointLight = new THREE.PointLight(0xffffff, 0.4);
            pointLight.position.set(-10, -10, -10);
            scene.add(pointLight);

            // Grid Floor
            const gridHelper = new THREE.GridHelper(30, 30, 0x334155, 0x1E293B);
            gridHelper.position.y = -2;
            scene.add(gridHelper);

            function createTextSprite(text, color = '#FFFFFF', fontSize = 44) {{
                const canvas = document.createElement('canvas');
                canvas.width = 256;
                canvas.height = 128;
                const ctx = canvas.getContext('2d');
                ctx.font = `Bold ${{fontSize}}px sans-serif`;
                ctx.fillStyle = color;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(text, 128, 64);

                const texture = new THREE.CanvasTexture(canvas);
                const spriteMaterial = new THREE.SpriteMaterial({{ map: texture, transparent: true }});
                const sprite = new THREE.Sprite(spriteMaterial);
                sprite.scale.set(3, 1.5, 1);
                return sprite;
            }}

            // SERVERS
            const serverGeo = new THREE.BoxGeometry(2.4, 1.4, 1.4);
            const serverMat = new THREE.MeshStandardMaterial({{
                color: 0x0284C7,
                metalness: 0.6,
                roughness: 0.2
            }});

            const server1 = new THREE.Mesh(serverGeo, serverMat);
            server1.position.set(-6, 3, 0);
            scene.add(server1);
            const s1Label = createTextSprite("SERVER 1", "#38BDF8", 38);
            s1Label.position.set(-6, 4.3, 0);
            scene.add(s1Label);

            const server2 = new THREE.Mesh(serverGeo, serverMat);
            server2.position.set(6, 3, 0);
            scene.add(server2);
            const s2Label = createTextSprite("SERVER 2", "#38BDF8", 38);
            s2Label.position.set(6, 4.3, 0);
            scene.add(s2Label);

            const interactiveObjects = [];

            function createConnectionLine(startPos, endPos, colorHexStr) {{
                const points = [startPos, endPos];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const colorHex = parseInt(colorHexStr.replace("#", "0x"));
                const material = new THREE.LineBasicMaterial({{
                    color: colorHex,
                    linewidth: 2,
                    transparent: true,
                    opacity: 0.65
                }});
                return new THREE.Line(geometry, material);
            }}

            const s1NodeIds = ["N1", "N2", "N3", "N4", "N5"];
            const s2NodeIds = ["N6", "N7", "N8", "N9", "N10"];

            s1NodeIds.forEach((nid, idx) => {{
                const x = -9 + idx * 1.5;
                const y = -0.5;
                const z = (idx % 2 === 0 ? 0.8 : -0.8);
                const nodePos = new THREE.Vector3(x, y, z);

                const tData = nodeData[nid] || {{}};
                const colorHexStr = tData.colorHex || "#64748B";
                const hexColor = parseInt(colorHexStr.replace("#", "0x"));
                
                const nodeGeo = new THREE.SphereGeometry(0.55, 32, 32);
                const nodeMat = new THREE.MeshStandardMaterial({{
                    color: hexColor,
                    metalness: 0.3,
                    roughness: 0.3,
                    emissive: hexColor,
                    emissiveIntensity: selectedNodes.includes(nid) ? 0.5 : 0.15
                }});

                const nodeMesh = new THREE.Mesh(nodeGeo, nodeMat);
                nodeMesh.position.copy(nodePos);
                nodeMesh.userData = {{ id: nid, data: tData }};
                scene.add(nodeMesh);
                interactiveObjects.push(nodeMesh);

                if (selectedNodes.includes(nid)) {{
                    const ringGeo = new THREE.RingGeometry(0.75, 0.85, 32);
                    const ringMat = new THREE.MeshBasicMaterial({{ color: 0xFFD700, side: THREE.DoubleSide }});
                    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
                    ringMesh.rotation.x = Math.PI / 2;
                    ringMesh.position.copy(nodePos);
                    scene.add(ringMesh);
                }}

                const label = createTextSprite(nid, colorHexStr, 44);
                label.position.set(x, y - 1.0, z);
                scene.add(label);

                const conn = createConnectionLine(server1.position, nodePos, colorHexStr);
                scene.add(conn);
            }});

            s2NodeIds.forEach((nid, idx) => {{
                const x = 3 + idx * 1.5;
                const y = -0.5;
                const z = (idx % 2 === 0 ? 0.8 : -0.8);
                const nodePos = new THREE.Vector3(x, y, z);

                const tData = nodeData[nid] || {{}};
                const colorHexStr = tData.colorHex || "#64748B";
                const hexColor = parseInt(colorHexStr.replace("#", "0x"));
                
                const nodeGeo = new THREE.SphereGeometry(0.55, 32, 32);
                const nodeMat = new THREE.MeshStandardMaterial({{
                    color: hexColor,
                    metalness: 0.3,
                    roughness: 0.3,
                    emissive: hexColor,
                    emissiveIntensity: selectedNodes.includes(nid) ? 0.5 : 0.15
                }});

                const nodeMesh = new THREE.Mesh(nodeGeo, nodeMat);
                nodeMesh.position.copy(nodePos);
                nodeMesh.userData = {{ id: nid, data: tData }};
                scene.add(nodeMesh);
                interactiveObjects.push(nodeMesh);

                if (selectedNodes.includes(nid)) {{
                    const ringGeo = new THREE.RingGeometry(0.75, 0.85, 32);
                    const ringMat = new THREE.MeshBasicMaterial({{ color: 0xFFD700, side: THREE.DoubleSide }});
                    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
                    ringMesh.rotation.x = Math.PI / 2;
                    ringMesh.position.copy(nodePos);
                    scene.add(ringMesh);
                }}

                const label = createTextSprite(nid, colorHexStr, 44);
                label.position.set(x, y - 1.0, z);
                scene.add(label);

                const conn = createConnectionLine(server2.position, nodePos, colorHexStr);
                scene.add(conn);
            }});

            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();

            window.addEventListener('mousemove', (event) => {{
                const rect = renderer.domElement.getBoundingClientRect();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(interactiveObjects);

                if (intersects.length > 0) {{
                    const hit = intersects[0].object;
                    const d = hit.userData.data;
                    const nid = hit.userData.id;
                    const isTargetNode = selectedNodes.includes(nid);

                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX - rect.left + 12) + 'px';
                    tooltip.style.top = (event.clientY - rect.top - 10) + 'px';

                    const deliveryHtml = isTargetNode 
                        ? `<div>Delivery: <b style="color:${{d.colorHex}};">${{d.delivery_status}}</b></div>`
                        : `<div style="color:#64748B;">Delivery: <i>N/A (not targeted)</i></div>`;

                    tooltip.innerHTML = `
                        <div class="tt-title">📱 Node ${{nid}} ${{isTargetNode ? '🎯 [TARGET]' : ''}}</div>
                        <div>Status: <span class="tt-badge" style="background:${{d.colorHex}}; color:#000;">${{d.stateLabel}}</span></div>
                        <div style="margin-top:4px;">SNR: <b>${{d.snr_db.toFixed(2)}} dB</b></div>
                        <div>Latency: <b>${{d.latency_ms.toFixed(2)}} ms</b></div>
                        <div>Packet Loss: <b>${{d.packet_loss_pct.toFixed(2)}}%</b></div>
                        ${{deliveryHtml}}
                    `;
                    document.body.style.cursor = 'pointer';
                }} else {{
                    tooltip.style.display = 'none';
                    document.body.style.cursor = 'default';
                }}
            }});

            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});

            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                server1.rotation.y += 0.005;
                server2.rotation.y += 0.005;
                renderer.render(scene, camera);
            }}
            animate();
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=height)
