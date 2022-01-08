import plotly.graph_objects as go


class Graph_Plotly:
    "Class to Create Graphs using the MongoDB Database"

    def __init__(self, database):
        self.database = database  #MongoDB Database object

    def create_graph(self):
        "Function to create a HTML report Document using Plotly"
        data = self.database.list_collection_names()
        dates = [file.split('_')[1] for file in data]
        d = dict()
        for i in range(len(dates)):
            d[dates[i]] = self.database[data[i]].estimated_document_count()

        x = list(d.keys())
        y = list(d.values())
        absent = [max(y) - i for i in y]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=x, y=y, name="Present", marker_color="rgb(11, 137, 43)"))
        fig.add_trace(
            go.Bar(x=x, y=absent, name="Absent", marker_color='indianred'))
        fig.update_layout(barmode='group',
                          xaxis_tickangle=-70,
                          xaxis_tickfont_size=10,
                          xaxis_title="Dates",
                          yaxis_title="Total Employee's",
                          legend_title="Attendance",
                          xaxis=dict(tickmode='linear'))
        fig.write_html(
            "dashboard\\dashboard_templates\\dashboard\\report.html")
