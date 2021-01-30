package com.example.amadeus;

import android.media.Image;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.List;

public class ChatAdapter extends RecyclerView.Adapter<ChatAdapter.MyViewHolder> {
    private ArrayList<ChatDTO> mDataset;
    // Provide a reference to the views for each data item
    // Complex data items may need more than one view per item, and
    // you provide access to all the views for a data item in a view holder
    public static class MyViewHolder extends RecyclerView.ViewHolder {
        // each data item is just a string in this case
        public View recyclerView;
        public MyViewHolder(View v) {
            super(v);
            recyclerView = v;
        }
    }
    public ChatAdapter(ArrayList<ChatDTO> myDataset) {
        mDataset = myDataset;
    }
    // Create new views (invoked by the layout manager)
    @Override
    public ChatAdapter.MyViewHolder onCreateViewHolder(ViewGroup parent,
                                                       int viewType) {
        // create a new view
        View v = (View) LayoutInflater.from(parent.getContext()).inflate(R.layout.chatting_item, parent, false);
        MyViewHolder vh = new MyViewHolder(v);
        return vh;
    }

    // Replace the contents of a view (invoked by the layout manager)
    @Override
    public void onBindViewHolder(MyViewHolder holder, int position) {
        // - get element from your dataset at this position
        // - replace the contents of the view with that element
        TextView name = (TextView) holder.recyclerView.findViewById(R.id.chat_name);
        TextView content = (TextView) holder.recyclerView.findViewById(R.id.chat_content);
        ImageView profile_image = (ImageView) holder.recyclerView.findViewById(R.id.profile_image);

        TextView my_name = (TextView) holder.recyclerView.findViewById(R.id.my_chat_name);
        TextView my_content = (TextView) holder.recyclerView.findViewById(R.id.my_chat_content);
        ImageView my_profile_image = (ImageView) holder.recyclerView.findViewById(R.id.my_profile_image);

        if(mDataset.get(position).name == "Okabe") {
            my_name.setText(mDataset.get(position).name);
            my_content.setText(mDataset.get(position).content);

            name.setVisibility(View.GONE);
            content.setVisibility(View.GONE);
            profile_image.setVisibility(View.GONE);
        }
        else{
            if(mDataset.get(position).name.equals("Daru"))
                profile_image.setImageResource(R.drawable.daru);
            else if(mDataset.get(position).name.equals("Mayushi"))
                profile_image.setImageResource(R.drawable.mayushi);
            name.setText(mDataset.get(position).name);
            content.setText(mDataset.get(position).content);

            my_name.setVisibility(View.GONE);
            my_content.setVisibility(View.GONE);
            my_profile_image.setVisibility(View.GONE);
        }
    }

    // Return the size of your dataset (invoked by the layout manager)
    @Override
    public int getItemCount() {
        return mDataset.size();
    }
}
