import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import networkx as nx
    import polars as pl
    import pyvis
    import pycountry
    import os
    return mo, nx, os, pl, pycountry, pyvis


@app.cell
def _(mo):
    mo.md(
        r"""
    # Network analysis
    Construct a network of trade relationships through time.

    Use this network to study diversion, and lay down some facts on the UK's position in the trade world.
    """
    )
    return


@app.cell
def _(pl):
    file_path = "data/final/unified_filtered_10000minval_top100countries/"
    unified_lf = pl.scan_parquet(file_path)

    print("Length of df:", unified_lf.select(pl.len()).collect().item())
    unified_lf.head().collect()
    return (unified_lf,)


@app.cell
def _(unified_lf):
    unified_lf.collect_schema()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Methodology
    A basic trade network is a collection of nodes and edges, cosntructed using the unified_lf dataset.

    This means doing the following:

    1. Group to the right level of aggregation for analysis
    2. For each year, create a multi digraph network
    3. Fill in this network using the aggregated trade values / volumes, creating a unique edge for each product code at that level of aggregation between trading partners.
    4. Store each of these matrices through time
    """
    )
    return


@app.cell
def _(unified_lf):
    unique_years = unified_lf.select("year").unique().collect()["year"].to_list()
    print(f"Unique years: {unique_years}")

    product_disaggregation_level = "2-digit"  # 6-digit, 4-digit, 2-digit, total
    return product_disaggregation_level, unique_years


@app.cell
def _(product_disaggregation_level):
    if product_disaggregation_level == "6-digit":
        product_code_len = 6
    if product_disaggregation_level == "4-digit":
        product_code_len = 4
    if product_disaggregation_level == "2-digit":
        product_code_len = 2
    if product_disaggregation_level == "total":
        product_code_len = None
    return (product_code_len,)


@app.cell
def _(mo, nx, pl, product_code_len, pycountry, unified_lf, unique_years):
    for year in unique_years:
        ### --- 1. Filter the LF to the right year
        filtered_lf = unified_lf.filter(pl.col("year") == year)
        print("Filtered for year:\n", filtered_lf.head().collect())

        ### --- 2. Group the data to the right level of aggregation
        if product_code_len:
            # Shorten the product codes
            filtered_lf = filtered_lf.with_columns(
                pl.col("product_code").str.slice(offset=0, length=product_code_len)
            )

            aggregated_lf = filtered_lf.group_by(
                ["reporter_country", "partner_country", "product_code"]
            ).agg(
                (
                    (pl.col("effective_tariff") * pl.col("value")).sum()
                    / pl.col("value").sum()
                ).alias("weighted_tariff"),
                pl.sum("value"),
                pl.sum("quantity"),
            )

        else:
            # Ignore product code - group across the rest
            aggregated_lf = filtered_lf.group_by(
                ["reporter_country", "partner_country"]
            ).agg(
                (
                    (pl.col("effective_tariff") * pl.col("value")).sum()
                    / pl.col("value").sum()
                ).alias("weighted_tariff"),
                pl.sum("value"),
                pl.sum("quantity"),
            )

        print("Collecting aggregated and filtered lf:")
        aggregated_df = aggregated_lf.collect(engine="streaming")
        print(aggregated_df.head())

        G_year = nx.MultiDiGraph()

        # Fill in the graph by iterating
        for row in mo.status.progress_bar(
            aggregated_df.iter_rows(named=True),
            title=f"Creating graph for year {year}",
            total=aggregated_df.height,
        ):
            reporter = row["reporter_country"]
            reporter_name = pycountry.countries.get(numeric=reporter).name

            partner = row["partner_country"]
            partner_name = pycountry.countries.get(numeric=partner).name

            # Add nodes (NetworkX handles duplicates automatically)
            G_year.add_node(reporter, label=reporter_name)
            G_year.add_node(partner, label=partner_name)

            attributes = {
                "value": row["value"],
                "quantity": row["quantity"],
                "weighted_tariff": row["weighted_tariff"],
                "year": year,  # Store year on edge too, can be useful
            }

            edge_key = None
            if (
                "product_code" in row
            ):  # This field exists if product_code_len was set
                attributes["product_code"] = row["product_code"]
                edge_key = row[
                    "product_code"
                ]  # Use aggregated product code as edge key
                G_year.add_edge(reporter, partner, key=edge_key, **attributes)
            else:
                # For MultiDiGraph, add_edge without a key will add a new edge with an auto key (e.g., 0)
                G_year.add_edge(reporter, partner, **attributes)

        break
    return G_year, year


@app.cell
def _():
    options_json_string = """
    {
      "nodes": {
        "font": {
          "size": 12
        },
        "size": 20
      },
      "edges": {
        "smooth": {
          "enabled": true,
          "type": "dynamic"
        },
        "scaling": {
          "min": 1,
          "max": 15,
          "label": { "enabled": false }
        },
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 0.7 }
        }
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -20000,
          "centralGravity": 0.20,
          "springLength": 250,
          "springConstant": 0.05,
          "damping": 0.75,
          "avoidOverlap": 0.3
        },
        "solver": "barnesHut",
        "stabilization": {
          "enabled": true,
          "iterations": 1500,
          "updateInterval": 25,
          "onlyDynamicEdges": false,
          "fit": true
        }
      },
      "interaction": {
        "hover": true,
        "hoverConnectedEdges": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": true,
        "selectable": true,
        "selectConnectedEdges": true,
        "multiselect": false
      },
      "manipulation": {
         "enabled": false
      }
    }
    """
    return (options_json_string,)


@app.cell
def _():
    custom_js_to_inject = """
    <script type="text/javascript">
        // Flag to ensure core setup logic runs only once
        if (typeof window.selectionInfoInitialized === 'undefined') {
            window.selectionInfoInitialized = false;
        }
        // Store all original node and edge data for resetting focus
        var originalNodesData = null;
        var originalEdgesData = null;
        var isFocused = false; // To track if the network is in a focused state

        function setupSelectionInfo() {
            if (typeof network !== 'undefined' && typeof nodes !== 'undefined' && typeof edges !== 'undefined') {
                if (window.selectionInfoInitialized) {
                    // console.log("Selection info setup already run.");
                    return;
                }
                // console.log("Running setupSelectionInfo for the first time.");

                // Store original data if not already stored
                if (!originalNodesData) {
                    originalNodesData = nodes.get({ returnType: "Array" });
                }
                if (!originalEdgesData) {
                    originalEdgesData = edges.get({ returnType: "Array" });
                }

                var infoDiv = document.getElementById('selectionInfo');
                // ... (infoDiv creation and styling - same as before) ...
                if (!infoDiv) {
                    infoDiv = document.createElement('div');
                    infoDiv.id = 'selectionInfo';
                    infoDiv.style.position = 'fixed'; infoDiv.style.top = '20px'; infoDiv.style.right = '20px';
                    infoDiv.style.width = 'auto'; infoDiv.style.minWidth = '250px'; infoDiv.style.maxWidth = '350px';
                    infoDiv.style.maxHeight = 'calc(100vh - 40px)'; infoDiv.style.overflowY = 'auto';
                    infoDiv.style.padding = '15px'; infoDiv.style.backgroundColor = '#ffffff';
                    infoDiv.style.border = '1px solid #dddddd'; infoDiv.style.borderRadius = '8px';
                    infoDiv.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)'; infoDiv.style.zIndex = '1000';
                    infoDiv.style.fontFamily = 'Arial, sans-serif'; infoDiv.style.fontSize = '13px';
                    infoDiv.style.lineHeight = '1.6';
                    document.body.appendChild(infoDiv);
                }
                var initialMessage = '<h5 style="margin-top:0; margin-bottom:10px; font-size:16px; color:#333; border-bottom:1px solid #eee; padding-bottom:8px;">Selection Details</h5><p style="font-size:0.9em; color:#777;">Click a node or an edge to see its information. <br>Double-click a node to focus on its neighborhood.</p>';
                infoDiv.innerHTML = initialMessage; 

                function formatKey(key) { /* ... (same as before) ... */ return key.replace(/_/g, ' ').replace(/\\b\\w/g, function(char) { return char.toUpperCase(); }); }
                function displayInfo(title, dataObject, isNodeFlag) { /* ... (same as before, ensure console logs are minimal if not debugging) ... */ 
                    // console.log(">>>> displayInfo called. Title: '" + title + "', isNodeFlag: " + isNodeFlag);
                    var content = '<h5 style="margin-top:0; margin-bottom:10px; font-size:16px; color:#333; border-bottom:1px solid #eee; padding-bottom:8px;">' + title + '</h5>';
                    content += '<ul style="list-style-type:none; padding-left:0; margin-bottom:0;">';
                    var keysToShow;
                    if (isNodeFlag === true) { keysToShow = Object.keys(dataObject); } else {
                        var preferredOrder = ['from', 'to', 'label', 'title', 'value', 'trade_value', 'weighted_tariff', 'raw_tariff', 'product_code', 'year', 'quantity'];
                        keysToShow = preferredOrder.filter(k => dataObject.hasOwnProperty(k)).concat(Object.keys(dataObject).filter(k => !preferredOrder.includes(k) && k !== 'id'));
                    }
                    for (var i = 0; i < keysToShow.length; i++) {
                        var key = keysToShow[i];
                        if (dataObject.hasOwnProperty(key)) {
                            if (isNodeFlag === true && (key === 'edges' || key === 'connections')) { continue; }
                            if (key === 'x' || key === 'y' || key === 'fixed' || key === 'physics' || key === 'hidden' || key === 'group' || key === 'level' || key === 'shape' || key === 'color' || key === 'font' || (key === 'value' && isNodeFlag && dataObject[key] === dataObject['size'])) { continue; }
                            if (key.startsWith('_')) { continue; }
                            var val = dataObject[key]; var formattedVal = String(val);
                            if (val === null || val === undefined) { formattedVal = '<em>N/A</em>'; }
                            else if (key === 'title' && typeof val === 'string') { formattedVal = '<pre style="white-space: pre-wrap; margin:0; padding: 3px 5px; background-color: #f0f0f0; border-radius:3px; font-size:0.9em;">' + val.replace(/\\n/g, '<br>') + '</pre>'; }
                            else if ((key === 'raw_tariff' || key === 'weighted_tariff') && typeof val === 'number') { formattedVal = (val * 100).toFixed(2) + '%'; }
                            else if ((key === 'value' || key === 'trade_value' || key === 'quantity') && typeof val === 'number') { formattedVal = val.toLocaleString(); }
                            else if (typeof val === 'object') { formattedVal = '<pre style="white-space: pre-wrap; margin:0; padding: 3px 5px; background-color: #f0f0f0; border-radius:3px; font-size:0.9em;">' + JSON.stringify(val, null, 2) + '</pre>';}
                            content += '<li style="margin-bottom:6px; padding-bottom:6px; border-bottom: 1px dashed #f0f0f0;"><strong style="color:#555;">' + formatKey(key) + ':</strong> ' + formattedVal + '</li>';
                        }
                    }
                    var lastBorderMarker = 'border-bottom: 1px dashed #f0f0f0;'; var lastLiEnd = content.lastIndexOf('</li>');
                    if (lastLiEnd > 0) { var lastLiStart = content.lastIndexOf('<li', lastLiEnd); if (lastLiStart > -1) { var tempSubstr = content.substring(lastLiStart, lastLiEnd); var lastBorderInLastLi = tempSubstr.indexOf(lastBorderMarker); if (lastBorderInLastLi !== -1) { content = content.substring(0, lastLiStart + lastBorderInLastLi) + tempSubstr.substring(lastBorderInLastLi + lastBorderMarker.length) + content.substring(lastLiEnd);}}}
                    content += '</ul>'; infoDiv.innerHTML = content;
                }
            
                function clearInfo() { infoDiv.innerHTML = initialMessage; }

                // --- Focus Logic ---
                function focusOnNode(nodeId) {
                    isFocused = true;
                    var neighborhoodNodesIds = [nodeId];
                    var neighborhoodEdgesIds = [];

                    var connectedEdges = network.getConnectedEdges(nodeId);
                    neighborhoodEdgesIds = neighborhoodEdgesIds.concat(connectedEdges);

                    connectedEdges.forEach(function(edgeId) {
                        var edge = edges.get(edgeId);
                        if (edge.from !== nodeId && !neighborhoodNodesIds.includes(edge.from)) {
                            neighborhoodNodesIds.push(edge.from);
                        }
                        if (edge.to !== nodeId && !neighborhoodNodesIds.includes(edge.to)) {
                            neighborhoodNodesIds.push(edge.to);
                        }
                    });
                
                    // Optionally, include edges between the neighbors themselves
                    // This can make the neighborhood graph very dense quickly.
                    // For now, we'll stick to edges connected to the central node.
                    // To add inter-neighbor edges:
                    // var interNeighborEdges = [];
                    // originalEdgesData.forEach(function(edge){
                    //    if (neighborhoodNodesIds.includes(edge.from) && neighborhoodNodesIds.includes(edge.to) && !neighborhoodEdgesIds.includes(edge.id)){
                    //        interNeighborEdges.push(edge.id);
                    //    }
                    // });
                    // neighborhoodEdgesIds = neighborhoodEdgesIds.concat(interNeighborEdges);


                    var updatedNodes = originalNodesData.map(function(node) {
                        node.hidden = !neighborhoodNodesIds.includes(node.id);
                        return node;
                    });

                    var updatedEdges = originalEdgesData.map(function(edge) {
                        // Show edges if they are in our neighborhood list
                        edge.hidden = !neighborhoodEdgesIds.includes(edge.id);
                        // Alternative: show edge if both its ends are in neighborhoodNodesIds
                        // edge.hidden = !(neighborhoodNodesIds.includes(edge.from) && neighborhoodNodesIds.includes(edge.to));
                        return edge;
                    });
                
                    nodes.clear(); // Clear current nodes DataSet
                    nodes.add(updatedNodes); // Add filtered nodes
                    edges.clear(); // Clear current edges DataSet
                    edges.add(updatedEdges); // Add filtered edges

                    // network.setData({ nodes: new vis.DataSet(updatedNodes), edges: new vis.DataSet(updatedEdges) }); // This also works but might reset layout
                    network.fit(); // Fit view to the focused neighborhood
                    console.log("Focused on node " + nodeId + " and its neighborhood.");
                }

                function resetFocus() {
                    if (!isFocused) return; // Do nothing if not focused
                
                    // Restore all original nodes and edges
                    // Ensure hidden property is removed or set to false
                    var restoredNodes = originalNodesData.map(function(node) {
                        delete node.hidden; // Or node.hidden = false;
                        return node;
                    });
                    var restoredEdges = originalEdgesData.map(function(edge) {
                        delete edge.hidden; // Or edge.hidden = false;
                        return edge;
                    });

                    nodes.clear();
                    nodes.add(restoredNodes);
                    edges.clear();
                    edges.add(restoredEdges);
                
                    isFocused = false;
                    network.fit(); // Fit view to the whole graph
                    clearInfo(); // Clear selection info as well
                    console.log("Network focus reset to full view.");
                }


                // --- Event Handlers ---
                network.off('selectNode'); network.off('selectEdge'); network.off('deselectNode'); network.off('deselectEdge'); network.off('click'); network.off('doubleClick');

                network.on('selectNode', function(nodeParams) {
                    // console.log("EVENT: selectNode. Params:", nodeParams);
                    if (nodeParams.nodes.length > 0) {
                        var selectedNodeId = nodeParams.nodes[0];
                        var selectedNodeData = nodes.get(selectedNodeId); 
                        if (!selectedNodeData) return; 
                        var isActuallyNode = true;
                        displayInfo('Selected Node (Double-click to focus)', selectedNodeData, isActuallyNode);
                    }
                });

                network.on('selectEdge', function(edgeParams) {
                    // console.log("EVENT: selectEdge. Params:", edgeParams);
                    if (edgeParams.nodes && edgeParams.nodes.length > 0 && !isFocused) { // If focused, allow edge selection within focus
                        // console.log("selectEdge: Event also involves selected node(s). Node info should be displayed.");
                        return; 
                    }
                    if (edgeParams.edges.length > 0) {
                        var selectedEdgeId = edgeParams.edges[0];
                        var selectedEdgeData = edges.get(selectedEdgeId);
                        if (!selectedEdgeData) return;
                        var isActuallyNode = false;
                        displayInfo('Selected Edge', selectedEdgeData, isActuallyNode);
                    }
                });
            
                network.on('doubleClick', function(params) {
                    console.log("EVENT: doubleClick. Params:", params);
                    if (params.nodes && params.nodes.length > 0) {
                        var nodeId = params.nodes[0];
                        focusOnNode(nodeId);
                    } else if (isFocused) { 
                        // If double-clicked on empty space while focused, reset focus
                        resetFocus();
                    }
                });

                network.on('click', function (clickParams) {
                    if (clickParams.nodes.length === 0 && clickParams.edges.length === 0) {
                        if (isFocused) {
                            // If clicked on empty space while focused, reset focus.
                            // Alternatively, could use a dedicated button.
                            // resetFocus(); // Decided to use double click on canvas to reset.
                        }
                        clearInfo(); // Clear info box on canvas click
                    }
                });
            
                network.on('deselectNode', function(params){
                    // If network is not in focused state, clear info.
                    // If it is focused, the focus remains until explicitly reset.
                    // The focusOnNode itself implies selection of the central node.
                    if(!isFocused){
                        clearInfo();
                    }
                });
                network.on('deselectEdge', function(params){
                     if(!isFocused){
                        clearInfo();
                    }
                });


                window.selectionInfoInitialized = true;
                // console.log("Selection info setup complete.");

            } else {
                // console.warn('VisJS Network, nodes, or edges not ready. Retrying setupSelectionInfo.');
                setTimeout(setupSelectionInfo, 200);
            }
        }

        if (document.readyState === "complete" || document.readyState === "interactive") {
            setupSelectionInfo();
        } else {
            document.addEventListener("DOMContentLoaded", function() {
                setupSelectionInfo();
            });
        }
    </script>
    """
    return (custom_js_to_inject,)


@app.cell(hide_code=True)
def _():
    # custom_js_to_inject = """
    # <script type="text/javascript">
    #     // Flag to ensure core setup logic runs only once
    #     if (typeof window.selectionInfoInitialized === 'undefined') {
    #         window.selectionInfoInitialized = false;
    #     }

    #     function setupSelectionInfo() {
    #         if (typeof network !== 'undefined' && typeof nodes !== 'undefined' && typeof edges !== 'undefined') {
    #             if (window.selectionInfoInitialized) {
    #                 console.log("Selection info setup already run. Skipping re-initialization.");
    #                 return;
    #             }
    #             console.log("Running setupSelectionInfo for the first time.");

    #             var infoDiv = document.getElementById('selectionInfo');
    #             if (!infoDiv) {
    #                 infoDiv = document.createElement('div');
    #                 infoDiv.id = 'selectionInfo';
    #                 // Styling (same as before)
    #                 infoDiv.style.position = 'fixed';
    #                 infoDiv.style.top = '20px';
    #                 infoDiv.style.right = '20px';
    #                 infoDiv.style.width = 'auto';
    #                 infoDiv.style.minWidth = '250px';
    #                 infoDiv.style.maxWidth = '350px';
    #                 infoDiv.style.maxHeight = 'calc(100vh - 40px)';
    #                 infoDiv.style.overflowY = 'auto';
    #                 infoDiv.style.padding = '15px';
    #                 infoDiv.style.backgroundColor = '#ffffff';
    #                 infoDiv.style.border = '1px solid #dddddd';
    #                 infoDiv.style.borderRadius = '8px';
    #                 infoDiv.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
    #                 infoDiv.style.zIndex = '1000';
    #                 infoDiv.style.fontFamily = 'Arial, sans-serif';
    #                 infoDiv.style.fontSize = '13px';
    #                 infoDiv.style.lineHeight = '1.6';
    #                 document.body.appendChild(infoDiv);
    #             }

    #             var initialMessage = '<h5 style="margin-top:0; margin-bottom:10px; font-size:16px; color:#333; border-bottom:1px solid #eee; padding-bottom:8px;">Selection Details</h5><p style="font-size:0.9em; color:#777;">Click a node or an edge to see its information here.</p>';
    #             infoDiv.innerHTML = initialMessage;

    #             function formatKey(key) {
    #                 return key.replace(/_/g, ' ').replace(/\\b\\w/g, function(char) { return char.toUpperCase(); });
    #             }

    #             function displayInfo(title, dataObject, isNodeFlag) {
    #                 console.log(">>>> displayInfo called. Title: '" + title + "', isNodeFlag: " + isNodeFlag + " (type: " + typeof isNodeFlag + ")");
    #                 // console.log(">>>> dataObject received:", JSON.parse(JSON.stringify(dataObject))); // Can be verbose

    #                 var content = '<h5 style="margin-top:0; margin-bottom:10px; font-size:16px; color:#333; border-bottom:1px solid #eee; padding-bottom:8px;">' + title + '</h5>';
    #                 content += '<ul style="list-style-type:none; padding-left:0; margin-bottom:0;">';

    #                 var keysToShow;
    #                 if (isNodeFlag === true) {
    #                     console.log("displayInfo: Processing as NODE.");
    #                     keysToShow = Object.keys(dataObject);
    #                 } else {
    #                     console.log("displayInfo: Processing as EDGE.");
    #                     var preferredOrder = ['from', 'to', 'label', 'title', 'value', 'trade_value', 'weighted_tariff', 'raw_tariff', 'product_code', 'year', 'quantity'];
    #                     keysToShow = preferredOrder.filter(k => dataObject.hasOwnProperty(k))
    #                                     .concat(Object.keys(dataObject).filter(k => !preferredOrder.includes(k) && k !== 'id'));
    #                 }
    #                 // console.log("displayInfo: Keys to show:", keysToShow); // Can be verbose

    #                 for (var i = 0; i < keysToShow.length; i++) {
    #                     var key = keysToShow[i];
    #                     if (dataObject.hasOwnProperty(key)) {
    #                         if (isNodeFlag === true && (key === 'edges' || key === 'connections')) { continue; }
    #                         if (key === 'x' || key === 'y' || key === 'fixed' || key === 'physics' || key === 'hidden' || key === 'group' || key === 'level' || key === 'shape' || key === 'color' || key === 'font' || key === 'value' && isNodeFlag) { // Also skip 'value' for nodes if it's just size
    #                              if (key === 'value' && isNodeFlag && dataObject[key] === dataObject['size']) { /* console.log('Skipping node key (value as size):', key); */ continue; }
    #                              else if (key !== 'value') { /* console.log('Skipping generic VisJS key:', key); */ continue; } // Allow node 'value' if it's not size
    #                         }
    #                         if (key.startsWith('_')) { continue; }

    #                         var val = dataObject[key];
    #                         var formattedVal = String(val);

    #                         if (val === null || val === undefined) {
    #                             formattedVal = '<em>N/A</em>';
    #                         } else if (key === 'title' && typeof val === 'string') {
    #                             formattedVal = '<pre style="white-space: pre-wrap; margin:0; padding: 3px 5px; background-color: #f0f0f0; border-radius:3px; font-size:0.9em;">' + val.replace(/\\n/g, '<br>') + '</pre>';
    #                         } else if ((key === 'raw_tariff' || key === 'weighted_tariff') && typeof val === 'number') {
    #                             formattedVal = (val * 100).toFixed(2) + '%';
    #                         } else if ((key === 'value' || key === 'trade_value' || key === 'quantity') && typeof val === 'number') { // 'value' here is for edges or a specific node value, not size
    #                             formattedVal = val.toLocaleString();
    #                         } else if (typeof val === 'object') {
    #                             formattedVal = '<pre style="white-space: pre-wrap; margin:0; padding: 3px 5px; background-color: #f0f0f0; border-radius:3px; font-size:0.9em;">' + JSON.stringify(val, null, 2) + '</pre>';
    #                         }

    #                         content += '<li style="margin-bottom:6px; padding-bottom:6px; border-bottom: 1px dashed #f0f0f0;"><strong style="color:#555;">' + formatKey(key) + ':</strong> ' + formattedVal + '</li>';
    #                     }
    #                 }

    #                 var lastBorderMarker = 'border-bottom: 1px dashed #f0f0f0;';
    #                 var lastLiEnd = content.lastIndexOf('</li>');
    #                 if (lastLiEnd > 0) {
    #                     var lastLiStart = content.lastIndexOf('<li', lastLiEnd);
    #                     if (lastLiStart > -1) {
    #                         var tempSubstr = content.substring(lastLiStart, lastLiEnd);
    #                         var lastBorderInLastLi = tempSubstr.indexOf(lastBorderMarker);
    #                         if (lastBorderInLastLi !== -1) {
    #                             content = content.substring(0, lastLiStart + lastBorderInLastLi) + tempSubstr.substring(lastBorderInLastLi + lastBorderMarker.length) + content.substring(lastLiEnd);
    #                         }
    #                     }
    #                 }
    #                 content += '</ul>';
    #                 infoDiv.innerHTML = content;
    #             }

    #             network.off('selectNode');
    #             network.off('selectEdge');
    #             network.off('deselectNode');
    #             network.off('deselectEdge');
    #             network.off('click');

    #             network.on('selectNode', function(nodeParams) {
    #                 console.log("EVENT: selectNode. Params:", nodeParams);
    #                 if (nodeParams.nodes.length > 0) {
    #                     var selectedNodeId = nodeParams.nodes[0];
    #                     // console.log("selectNode: ID = " + selectedNodeId);
    #                     var selectedNodeData = nodes.get(selectedNodeId);
    #                     if (!selectedNodeData) { console.error("selectNode: Node data not found for ID:", selectedNodeId); return; }
    #                     // console.log("selectNode: Fetched Data:", JSON.parse(JSON.stringify(selectedNodeData)));
    #                     var isActuallyNode = true;
    #                     // console.log("selectNode: Calling displayInfo with isActuallyNode = " + isActuallyNode);
    #                     displayInfo('Selected Node', selectedNodeData, isActuallyNode);
    #                 }
    #             });

    #             network.on('selectEdge', function(edgeParams) {
    #                 console.log("EVENT: selectEdge. Params:", edgeParams);

    #                 // ** THE FIX IS HERE **
    #                 // If the 'selectEdge' event parameters also indicate that nodes were selected
    #                 // (i.e., edgeParams.nodes is not empty), it implies this edge selection might be
    #                 // secondary to a node click, or the click was ambiguous.
    #                 // In this case, we prioritize the node's info (which should have been handled by selectNode).
    #                 if (edgeParams.nodes && edgeParams.nodes.length > 0) {
    #                     console.log("selectEdge: Event also involves selected node(s):", edgeParams.nodes, ". Node info display should take precedence. Suppressing this edge info update.");
    #                     // Optionally, you could check if the node info DIV is already showing info for one of edgeParams.nodes.
    #                     // For now, simply returning will prevent this edge event from overwriting.
    #                     return;
    #                 }

    #                 // If we are here, it means edgeParams.nodes was empty, so it's a "pure" edge selection.
    #                 if (edgeParams.edges.length > 0) {
    #                     var selectedEdgeId = edgeParams.edges[0];
    #                     // console.log("selectEdge: ID = " + selectedEdgeId + " (pure edge selection).");
    #                     var selectedEdgeData = edges.get(selectedEdgeId);
    #                     if (!selectedEdgeData) { console.error("selectEdge: Edge data not found for ID:", selectedEdgeId); return; }
    #                     // console.log("selectEdge: Fetched Data:", JSON.parse(JSON.stringify(selectedEdgeData)));
    #                     var isActuallyNode = false;
    #                     // console.log("selectEdge: Calling displayInfo with isActuallyNode = " + isActuallyNode);
    #                     displayInfo('Selected Edge', selectedEdgeData, isActuallyNode);
    #                 } else {
    #                     // This case should be less common if the above check for edgeParams.nodes is active
    #                     console.log("selectEdge event, but params.edges is empty (and no nodes associated with this event).");
    #                 }
    #             });

    #             function clearInfo() {
    #                 // console.log("Clearing selection info box.");
    #                 infoDiv.innerHTML = initialMessage;
    #             }

    #             network.on('deselectNode', clearInfo);
    #             network.on('deselectEdge', clearInfo);
    #             network.on("click", function (clickParams) {
    #                 if (clickParams.nodes.length === 0 && clickParams.edges.length === 0) {
    #                     // console.log("EVENT: Canvas click (no nodes/edges).");
    #                     clearInfo();
    #                 }
    #             });

    #             window.selectionInfoInitialized = true;
    #             console.log("Selection info setup complete. Event handlers are active.");

    #         } else {
    #             console.warn('VisJS Network, nodes, or edges not ready. Retrying setupSelectionInfo in 200ms.');
    #             setTimeout(setupSelectionInfo, 200);
    #         }
    #     }

    #     if (document.readyState === "complete" || document.readyState === "interactive") {
    #         // console.log('DOM ready or interactive, calling setupSelectionInfo.');
    #         setupSelectionInfo();
    #     } else {
    #         document.addEventListener("DOMContentLoaded", function() {
    #             // console.log('DOMContentLoaded event fired, calling setupSelectionInfo.');
    #             setupSelectionInfo();
    #         });
    #     }
    # </script>
    # """
    return


@app.cell
def _(G_year, custom_js_to_inject, options_json_string, os, pyvis, year):
    base_path = "src/mpil_tariff_trade_analysis/dev2/network/vis/"
    temp_pyvis_filename = f"/temp_pyvis_output_for_year_{year}.html"
    final_pyvis_filename = f"/G_year_{year}_interactive_network.html"

    pyvis_G = pyvis.network.Network("750px", "750px")
    pyvis_G.from_nx(G_year)
    pyvis_G.set_options(options_json_string)

    # Write the temp file
    pyvis_G.write_html(
        base_path + temp_pyvis_filename,
        notebook=False,
    )

    # read the temp file's html
    html_content = ""
    try:
        with open(base_path + temp_pyvis_filename, "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        print(f"Error: Pyvis temporary file {temp_pyvis_filename} not found.")

    # Inject our html
    body_end_tag = "</body>"
    modified_html_content = html_content.replace(
        body_end_tag, custom_js_to_inject + "\n" + body_end_tag
    )

    # Save our file
    with open(base_path + final_pyvis_filename, "w", encoding="utf-8") as f:
        f.write(modified_html_content)

    if os.path.exists(base_path + temp_pyvis_filename):
        os.remove(base_path + temp_pyvis_filename)
    return


@app.cell
def _(G_year):
    G_year.nodes["818"]
    return


if __name__ == "__main__":
    app.run()
